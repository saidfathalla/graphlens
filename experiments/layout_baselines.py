#!/usr/bin/env python3
"""Layout-classifier baselines and feature ablation under leave-one-domain-out CV.

Reproduces the comparison table for the paper:
  python3 experiments/layout_baselines.py -t data/training_data.csv -o results/layout

Writes layout_baselines.csv and layout_per_domain.csv (plus layout_baselines.tex with --latex).
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

MODELS = {
    "Majority class": lambda s: DummyClassifier(strategy="most_frequent"),
    "Decision tree (GraphLens)": lambda s: DecisionTreeClassifier(
        criterion="entropy", max_depth=8, min_samples_leaf=2, random_state=s),
    "Random forest": lambda s: RandomForestClassifier(300, random_state=s),
    "Logistic regression": lambda s: LogisticRegression(max_iter=2000),
    "k-NN (k=5)": lambda s: KNeighborsClassifier(5),
    "Multinomial naive Bayes": lambda s: MultinomialNB(),
}
REF = ("Column names + shape", "Decision tree (GraphLens)")


def lodo_predict(X, y, groups, make):
  pred = np.empty(len(y), dtype=object)
  for tr, te in LeaveOneGroupOut().split(X, y, groups):
    vec = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    model = make().fit(vec.fit_transform(X.iloc[tr]), y[tr])
    pred[te] = model.predict(vec.transform(X.iloc[te]))
  return pred


def lookup_predict(X, y, groups):
  """Rule baseline without learning: most frequent layout per descriptor string."""
  pred = np.empty(len(y), dtype=object)
  for tr, te in LeaveOneGroupOut().split(X, y, groups):
    table = pd.Series(y[tr]).groupby(X.iloc[tr].values).agg(lambda s: s.mode()[0])
    fallback = pd.Series(y[tr]).mode()[0]
    pred[te] = [table.get(x, fallback) for x in X.iloc[te]]
  return pred


def bootstrap_ci(correct, n_boot, rng):
  idx = rng.integers(0, len(correct), (n_boot, len(correct)))
  return np.percentile(correct[idx].mean(axis=1), [2.5, 97.5])


def mcnemar_p(ref_correct, other_correct):
  b = int((ref_correct & ~other_correct).sum())
  c = int((~ref_correct & other_correct).sum())
  return 1.0 if b + c == 0 else binomtest(b, b + c, 0.5).pvalue


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("-t", "--training", default="data/training_data.csv")
  ap.add_argument("-o", "--output", default="results/layout")
  ap.add_argument("--seed", type=int, default=42)
  ap.add_argument("--n-boot", type=int, default=2000)
  ap.add_argument("--latex", action="store_true", help="also write a LaTeX table")
  a = ap.parse_args()
  os.makedirs(a.output, exist_ok=True)

  df = pd.read_csv(a.training).fillna("")
  y, groups = df["target_chart"].astype(str).values, df["domain"].astype(str).values
  feats = {
      "Column names + shape": (df["columns_text"] + " " + df["descriptor"]).str.strip(),
      "Shape descriptors only": df["descriptor"].astype(str),
      "Column names only": df["columns_text"].astype(str),
  }
  preds = {}
  for fname, X in feats.items():
    for mname, make in MODELS.items():
      preds[(fname, mname)] = lodo_predict(X, y, groups, lambda: make(a.seed))
  preds[("Shape descriptors only", "Lookup table (no learning)")] = lookup_predict(
      feats["Shape descriptors only"], y, groups)

  rng = np.random.default_rng(a.seed)
  ref_ok = preds[REF] == y
  rows = []
  for (fname, mname), p in preds.items():
    ok = p == y
    lo, hi = bootstrap_ci(ok, a.n_boot, rng)
    rows.append(dict(features=fname, model=mname,
                     accuracy=accuracy_score(y, p), ci95_low=lo, ci95_high=hi,
                     macro_f1=f1_score(y, p, average="macro"),
                     mcnemar_p_vs_graphlens=np.nan if (fname, mname) == REF else mcnemar_p(ref_ok, ok)))
  res = pd.DataFrame(rows)
  res.to_csv(os.path.join(a.output, "layout_baselines.csv"), index=False)

  per = pd.DataFrame({d: {f"{f} / {m}": float((p[groups == d] == y[groups == d]).mean())
                          for (f, m), p in preds.items()} for d in sorted(set(groups))})
  per.to_csv(os.path.join(a.output, "layout_per_domain.csv"))

  if a.latex:
   with open(os.path.join(a.output, "layout_baselines.tex"), "w", encoding="utf-8") as f:
    f.write("\\begin{tabular}{llccc}\n\\toprule\n"
            "Features & Model & Accuracy [95\\% CI] & Macro-F1 & McNemar $p$ \\\\\n\\midrule\n")
    for _, r in res.iterrows():
      pv = "--" if np.isnan(r.mcnemar_p_vs_graphlens) else f"{r.mcnemar_p_vs_graphlens:.3f}"
      f.write(f"{r.features} & {r.model} & {100*r.accuracy:.1f} [{100*r.ci95_low:.1f}, "
              f"{100*r.ci95_high:.1f}] & {r.macro_f1:.3f} & {pv} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

  print(f"n={len(y)} rows, {len(set(groups))} domains (leave-one-domain-out)\n")
  print(res.round(3).to_string(index=False))
  print(f"\nWritten to {a.output}/")


if __name__ == "__main__":
  main()
