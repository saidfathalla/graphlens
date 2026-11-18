#!/usr/bin/env python3
"""Masking experiment for the deterministic missing-value recovery (Module 1).

Observed values of the success count (x2) and derived ratio (y) are hidden at random
(MCAR) at several rates, the GraphLens normalisation is re-run on the masked input, and
the recovered values are compared with the hidden ground truth. Statistical imputers
(mean, median, k-NN, MICE) are run on the same masks as baselines.

  python3 experiments/recovery_masking.py -q queries/eventskg-query.rq \
      -f data/eventskg.ttl -p profiles/eventskg-profile.yaml -o results/recovery
"""
import argparse
import logging
import os
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graphlens import GraphLensAgnosticPipeline  # noqa: E402

warnings.filterwarnings("ignore")
SCENARIOS = ("mask_y", "mask_x2", "mask_both")


def load(a):
  out = tempfile.mkdtemp(prefix="gl_recovery_")
  pipe = GraphLensAgnosticPipeline(a.profile, a.training, out, a.seed)
  if a.query.startswith(("http://", "https://")):
    import requests
    q = requests.get(a.query, timeout=30).text
  else:
    q = open(a.query, encoding="utf-8").read()
  raw = pipe.query_to_dataframe(q, rdf_url=a.url, rdf_file=a.file)
  raw["_rid"] = np.arange(len(raw))
  return pipe, raw


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("-q", "--query", required=True)
  ap.add_argument("-p", "--profile", required=True)
  ap.add_argument("-t", "--training", default="data/training_data.csv")
  ap.add_argument("-o", "--output", default="results/recovery")
  g = ap.add_mutually_exclusive_group(required=True)
  g.add_argument("-u", "--url")
  g.add_argument("-f", "--file")
  ap.add_argument("--rates", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5])
  ap.add_argument("--repeats", type=int, default=5)
  ap.add_argument("--seed", type=int, default=42)
  ap.add_argument("--latex", action="store_true", help="also write a LaTeX table (30%% rate)")
  a = ap.parse_args()
  logging.basicConfig(level=logging.ERROR)
  os.makedirs(a.output, exist_ok=True)

  pipe, raw = load(a)
  x1, x2, y = pipe.k_vol, pipe.k_suc, pipe.k_rate
  inv = {v: k for k, v in pipe.var_map.items()}           # identifier -> SPARQL column
  col_x2, col_y = inv.get(x2), inv.get(y)
  if not (col_x2 in raw.columns and col_y in raw.columns):
    sys.exit(f"Profile must map both {x2!r} and {y!r} to SPARQL columns.")

  base = pipe.normalize(raw.copy())
  truth = base[base[[x1, x2, y]].notna().all(axis=1)
               & ~base[["y_recovered", "x2_recovered", "inconsistent"]].any(axis=1)]
  truth = truth.set_index("_rid")[[x1, x2, y, "year"]]
  ids = truth.index.values
  print(f"{len(raw)} raw rows; {len(truth)} fully observed rows used as ground truth")

  rows = []
  for rate in a.rates:
    for rep in range(a.repeats):
      rng = np.random.default_rng(a.seed + rep)
      for scen in SCENARIOS:
        hide_y = rng.random(len(ids)) < rate if scen in ("mask_y", "mask_both") else np.zeros(len(ids), bool)
        hide_x2 = rng.random(len(ids)) < rate if scen in ("mask_x2", "mask_both") else np.zeros(len(ids), bool)
        targets = [(y, ids[hide_y]), (x2, ids[hide_x2])]

        # GraphLens: mask the raw SPARQL projection and re-run its own normalisation
        rm = raw.copy()
        rm.loc[rm["_rid"].isin(ids[hide_y]), col_y] = np.nan
        rm.loc[rm["_rid"].isin(ids[hide_x2]), col_x2] = np.nan
        rec = pipe.normalize(rm).set_index("_rid")
        methods = {"GraphLens (algebraic)": rec}

        # Baselines: same masks applied to the clean normalised table
        masked = truth.copy()
        masked.loc[ids[hide_y], y] = np.nan
        masked.loc[ids[hide_x2], x2] = np.nan
        cols = [x1, x2, y, "year"]
        for name, imp in [("Mean", SimpleImputer(strategy="mean")),
                          ("Median", SimpleImputer(strategy="median")),
                          ("k-NN (k=5)", KNNImputer(n_neighbors=5)),
                          ("MICE", IterativeImputer(random_state=a.seed + rep, max_iter=20))]:
          filled = pd.DataFrame(imp.fit_transform(masked[cols]), index=masked.index, columns=cols)
          methods[name] = filled

        for var, hid in targets:
          if len(hid) == 0:
            continue
          true = truth.loc[hid, var]
          for name, frame in methods.items():
            est = frame.reindex(hid)[var]
            got = est.notna()
            err = (est[got] - true[got]).abs()
            rows.append(dict(rate=rate, repeat=rep, scenario=scen, variable=var, method=name,
                             n_hidden=len(hid), coverage=float(got.mean()),
                             mae=float(err.mean()) if got.any() else np.nan,
                             exact=float((err < (0.0005 if var == y else 0.5)).mean()) if got.any() else np.nan))

  res = pd.DataFrame(rows)
  res.to_csv(os.path.join(a.output, "recovery_masking_runs.csv"), index=False)
  summ = (res.groupby(["scenario", "variable", "rate", "method"])
          .agg(coverage=("coverage", "mean"), mae=("mae", "mean"), mae_sd=("mae", "std"),
               exact=("exact", "mean"))
          .reset_index())
  summ.to_csv(os.path.join(a.output, "recovery_masking_summary.csv"), index=False)

  at = summ[summ.rate == 0.3]
  if a.latex:
   with open(os.path.join(a.output, "recovery_masking.tex"), "w", encoding="utf-8") as f:
    f.write("\\begin{tabular}{lllccc}\n\\toprule\nScenario & Variable & Method & Coverage & MAE & Exact \\\\\n\\midrule\n")
    for _, r in at.iterrows():
      f.write(f"{r.scenario.replace('_', ' ')} & {r.variable} & {r.method} & {100*r.coverage:.1f}\\% & "
              f"{r.mae:.4f} & {100*r.exact:.1f}\\% \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
  pd.set_option("display.width", 160)
  print(at.round(4).to_string(index=False))
  print(f"\nWritten to {a.output}/")


if __name__ == "__main__":
  main()
