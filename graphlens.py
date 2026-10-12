#!/usr/bin/env python3
"""GraphLens v2 - Profile-configurable normalization, layout routing, clustering,

and longitudinal analytics for RDF/SPARQL result sets.
"""

import argparse
from contextlib import contextmanager
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from scipy import stats as sps
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
import yaml

logging.getLogger("rdflib.term").setLevel(logging.ERROR)
log = logging.getLogger("graphlens")

LAYOUTS = ["Line", "Bar", "Geomap", "Scatter", "Boxplot", "Violin", "Heatmap"]
YEAR_RE = re.compile(r"(?:19|20)\d{2}")


# --------------------------------------------------------------------------- helpers
def shannon_entropy(series, normalize=True):
  """Shannon entropy H(X) (Eq. 1) of a column's empirical value distribution."""
  p = series.dropna().value_counts(normalize=True).values
  if len(p) <= 1:
    return 0.0
  h = float(-(p * np.log2(p)).sum())
  return h / np.log2(len(p)) if normalize else h


def split_identifier(name):
  """camelCase / PascalCase / snake_case -> 'Camel Case' (preserves acronyms like CVPR)."""
  if pd.isna(name):
    return name
  s = str(name).strip()
  if s.isupper():
    return s
  s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s).replace("_", " ")
  return " ".join(
      w if w.isupper() else w.capitalize() for w in s.split()
  ).strip()


def describe_view(df, time_col=None, geo_col=None, lowcard_max=20):
  """Domain-agnostic shape tokens for a result table (input to the layout classifier)."""
  toks = []
  if time_col and time_col in df.columns:
    toks.append("__temporal")
  num = [
      c
      for c in df.columns
      if c != time_col and pd.api.types.is_numeric_dtype(df[c])
  ]
  cats = [c for c in df.columns if c != time_col and c not in num]
  toks.append(
      {0: "__numeric_0", 1: "__numeric_1", 2: "__numeric_2"}.get(
          len(num), "__numeric_3plus"
      )
  )
  for c in num:
    s = df[c].dropna()
    if len(s) and s.min() >= 0 and s.max() <= 1:
      toks.append("__bounded")
    if len(s) >= 8 and abs(sps.skew(s)) > 1.5:
      toks.append("__skewed")
  primary = next((c for c in cats if c != geo_col), None)
  if geo_col and geo_col in df.columns:
    toks.append("__geo")
    primary = primary or geo_col
  if primary:
    if geo_col is None or primary != geo_col:
      toks.append(
          "__lowcard"
          if df[primary].nunique() <= lowcard_max
          else "__highcard"
      )
    toks.append(
        "__one_row_per_key" if df[primary].is_unique else "__many_rows_per_key"
    )
  return " ".join(sorted(set(toks)))


# --------------------------------------------------------------------------- layout classifier
class LayoutClassifier:
  """Entropy (information-gain) decision tree over column-name tokens + shape tokens."""

  FALLBACK = [
      ("year submitted", "__temporal __numeric_1 __one_row_per_key", "Line"),
      ("acronym submitted", "__highcard __numeric_1 __one_row_per_key", "Bar"),
      ("country count", "__geo __numeric_1 __one_row_per_key", "Geomap"),
      ("submitted acceptance rate", "__numeric_2 __bounded __skewed", "Scatter"),
      (
          "field acceptance rate",
          "__lowcard __numeric_1 __many_rows_per_key __bounded",
          "Boxplot",
      ),
      (
          "cluster profile submitted",
          "__lowcard __numeric_1 __many_rows_per_key __skewed",
          "Violin",
      ),
      (
          "year submitted accepted acceptance rate",
          "__temporal __numeric_3plus",
          "Heatmap",
      ),
  ]

  def __init__(self, training_path=None, seed=42):
    self.vec = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    self.clf = DecisionTreeClassifier(
        criterion="entropy", max_depth=8, min_samples_leaf=2, random_state=seed
    )
    self.validation = {}
    self._fit(training_path)

  @staticmethod
  def _text(names, shape):
    return f"{' '.join(split_identifier(n) for n in names)} {shape}".strip()

  def _fit(self, path):
    if path and os.path.exists(path) and os.path.getsize(path) > 0:
      df = pd.read_csv(path).fillna("")
      if "descriptor" not in df:
        df["descriptor"] = ""
      X = (
          df["columns_text"].astype(str) + " " + df["descriptor"].astype(str)
      ).str.strip()
      y = df["target_chart"].astype(str)
      log.info("Loaded %d layout-training instances from %s", len(df), path)
      groups = df["domain"] if "domain" in df else None
    else:
      log.warning(
          "Training CSV missing -> built-in fallback (7 rows). Results will not"
          " generalise."
      )
      X = pd.Series([f"{a} {b}" for a, b, _ in self.FALLBACK])
      y = pd.Series([c for *_, c in self.FALLBACK])
      groups = None
    Xv = self.vec.fit_transform(X)
    self.clf.fit(Xv, y)
    self.classes_ = list(self.clf.classes_)
    if groups is not None and groups.nunique() >= 3:
      pred = cross_val_predict(
          DecisionTreeClassifier(
              criterion="entropy",
              max_depth=8,
              min_samples_leaf=2,
              random_state=42,
          ),
          Xv,
          y,
          groups=groups,
          cv=LeaveOneGroupOut(),
      )
      self.validation = {
          "cv": "leave-one-domain-out",
          "n": int(len(y)),
          "domains": int(groups.nunique()),
          "accuracy": float(accuracy_score(y, pred)),
          "macro_f1": float(f1_score(y, pred, average="macro")),
      }
      log.info("Layout classifier LODO-CV: %s", self.validation)

  def rank(self, names, shape):
    """Layouts ordered by tree confidence (predict_proba); never empty."""
    v = self.vec.transform([self._text(names, shape)])
    proba = self.clf.predict_proba(v)[0]
    order = np.argsort(-proba, kind="stable")
    return [(self.classes_[i], float(proba[i])) for i in order]


# --------------------------------------------------------------------------- views
@dataclass
class View:
  name: str
  df: pd.DataFrame
  title: str
  fname: str
  expected: str = ""
  time: str = None
  geo: str = None
  xlabel: str = ""
  ylabel: str = ""
  stacked: bool = False
  fits: dict = field(default_factory=dict)


class GraphLensAgnosticPipeline:

  def __init__(
      self,
      profile_path_or_url,
      training_data_path="data/training_data.csv",
      output_dir="output",
      seed=42,
  ):
    self.output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)
    self.seed = seed
    self.timings, self.report = {}, {}
    self._load_profile(profile_path_or_url)
    with self._timer("module2_bootstrap"):
      self.layout = LayoutClassifier(training_data_path, seed=seed)
    self.renderers = {
        "Line": self._r_line,
        "Bar": self._r_bar,
        "Geomap": self._r_geo,
        "Scatter": self._r_scatter,
        "Boxplot": self._r_box,
        "Violin": self._r_violin,
        "Heatmap": self._r_heat,
    }

  @contextmanager
  def _timer(self, name):
    t0 = time.perf_counter()
    yield
    self.timings[name] = self.timings.get(name, 0.0) + time.perf_counter() - t0

  def _load_profile(self, src):
    if src.startswith(("http://", "https://")):
      import requests

      r = requests.get(src, timeout=30)
      r.raise_for_status()
      prof = yaml.safe_load(r.text)
    else:
      if not os.path.exists(src):
        raise FileNotFoundError(f"Profile not found: {src}")
      with open(src, encoding="utf-8") as f:
        prof = yaml.safe_load(f)
    self.profile = prof
    lab = prof.get("labels", {})
    self.L = {
        k: lab.get(k, d)
        for k, d in dict(
            volume="Primary volume (x1)",
            success="Success count (x2)",
            rate="Derived ratio (y)",
            category="Category (c1)",
            location="Location (c2)",
            entity_identifier="Entity",
            entity_group="Entity group",
        ).items()
    }
    self.bands = prof.get(
        "cluster_bands",
        ["Low-ratio band", "Moderate-ratio band", "High-ratio band"],
    )
    cfg = prof.get("clustering", {})
    self.cl = dict(
        log_volume=cfg.get("log_volume", True),
        k_min=cfg.get("k_min", 2),
        k_max=cfg.get("k_max", 6),
        fixed_k=cfg.get("fixed_k"),
        stability_runs=cfg.get("stability_runs", 10),
    )
    self.filters = prof.get("filters", {})
    self.corr_method = prof.get("analysis", {}).get("corr_method", "spearman")
    self.aliases = prof.get("value_aliases", {})
    attrs = prof.get("attributes", {})
    self.var_map, self.display = {}, {}
    self.geo_ids = set()
    for group, items in attrs.items():
      for it in items:
        ident, sv = it.get("identifier"), it.get("sparql_var")
        if not (ident and sv):
          continue
        for s in sv if isinstance(sv, list) else [sv]:
          self.var_map[s] = ident
        self.display[ident] = sv[0] if isinstance(sv, list) else sv
        if it.get("semantic_type") == "geo":
          self.geo_ids.add(ident)
    ids = {
        g: [i["identifier"] for i in attrs.get(g, [])]
        for g in [
            "independent_metrics",
            "derived_ratios",
            "nominal_categories",
            "temporal_indicators",
            "entity_identifiers",
        ]
    }
    pick = lambda g, d, n=0: (ids[g][n] if len(ids[g]) > n else d)
    self.k_vol, self.k_suc = pick("independent_metrics", "x1"), pick(
        "independent_metrics", "x2", 1
    )
    self.k_rate, self.k_cat = pick("derived_ratios", "y"), pick(
        "nominal_categories", "c1"
    )
    self.k_geo = next(
        iter(self.geo_ids), pick("nominal_categories", "c2", 1)
    )
    self.k_time, self.k_ent = pick("temporal_indicators", "t"), pick(
        "entity_identifiers", "e"
    )
    log.info("Profile loaded (target class: %s)", prof.get("target_class", "n/a"))

  @staticmethod
  def _suffix(val):
    s = str(val).strip()
    return s.split("/")[-1].split("#")[-1] if ("/" in s or "#" in s) else s

  @staticmethod
  def _series_acronym(val):
    if pd.isna(val):
      return np.nan
    s = str(val).strip()
    out = re.sub(r"[\s_\-']*(?:19|20)\d{2}$", "", s).strip()
    return out.upper() if out else s.upper()

  def normalize(self, raw):
    rep = {"rows_raw": int(len(raw))}
    df = raw.rename(
        columns={k: v for k, v in self.var_map.items() if k in raw.columns}
    ).copy()

    potential_num_cols = {
        self.k_vol,
        self.k_suc,
        self.k_rate,
        "acceptanceRate",
        "submittedPapers",
        "acceptedPapers",
    }
    for c in potential_num_cols:
      if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    tcol = next(
        (c for c in (self.k_time, "date", "year") if c in df.columns), None
    )
    df["year"] = (
        df[tcol].map(
            lambda x: (
                int(YEAR_RE.search(str(x)).group())
                if pd.notna(x) and YEAR_RE.search(str(x))
                else np.nan
            )
        )
        if tcol
        else np.nan
    )

    for c in df.columns:
      if c not in ("year", tcol) and not pd.api.types.is_numeric_dtype(df[c]):
        df[c] = df[c].map(lambda x: self._suffix(x) if pd.notna(x) else np.nan)
    for ident, amap in self.aliases.items():
      if ident in df.columns:
        df[ident] = df[ident].replace(amap)
    for ident in (self.k_cat, self.k_geo):
      if ident in df.columns:
        df[ident] = df[ident].map(split_identifier)
    ent = (
        self.k_ent
        if self.k_ent in df.columns
        else next((c for c in ("acronym", "label") if c in df.columns), None)
    )
    if ent:
      df["series_acronym"] = df[ent].map(self._series_acronym)
      df["edition_label"] = [
          (
              f"{a} {int(y)}"
              if pd.notna(a) and pd.notna(y)
              else str(o)
          )
          for a, y, o in zip(df["series_acronym"], df["year"], df[ent])
      ]

    num = [c for c in (self.k_vol, self.k_suc, self.k_rate) if c in df.columns]
    rep["missing_before"] = {c: int(df[c].isna().sum()) for c in num}

    x1, x2, y = self.k_vol, self.k_suc, self.k_rate
    have = x1 in df.columns and x2 in df.columns and y in df.columns
    for f in ("y_rescaled", "y_recovered", "x2_recovered", "inconsistent"):
      df[f] = False
    if y in df.columns:
      m = df[y] > 1.0
      df.loc[m, y] = df.loc[m, y] / 100.0
      df.loc[m, "y_rescaled"] = True
      bad = (df[y] < 0) | (df[y] > 1)
      rep["y_out_of_range_set_missing"] = int(bad.sum())
      df.loc[bad, y] = np.nan
    if have:
      m = df[x2].notna() & df[x1].notna() & (df[x2] > df[x1])
      rep["x2_gt_x1_set_missing"] = int(m.sum())
      df.loc[m, x2] = np.nan
      df.loc[m, "inconsistent"] = True
      m = df[y].isna() & (df[x1] > 0) & df[x2].notna()
      df.loc[m, y] = df.loc[m, x2] / df.loc[m, x1]
      df.loc[m, "y_recovered"] = True
      m = df[x2].isna() & df[y].notna() & (df[x1] > 0)
      df.loc[m, x2] = np.round(df.loc[m, x1] * df.loc[m, y])
      df.loc[m, "x2_recovered"] = True
    rep["recovered"] = {
        f: int(df[f].sum())
        for f in (
            "y_rescaled",
            "y_recovered",
            "x2_recovered",
            "inconsistent",
        )
    }
    rep["missing_after"] = {c: int(df[c].isna().sum()) for c in num}

    n0 = len(df)
    df = df.dropna(subset=["year", x1]).copy()
    rep["dropped_missing_year_or_volume"] = int(n0 - len(df))
    df["year"] = df["year"].astype(int)
    lo, hi = self.filters.get("year_min"), self.filters.get("year_max")
    n1 = len(df)
    if lo is not None:
      df = df[df["year"] >= lo]
    if hi is not None:
      df = df[df["year"] <= hi]
    rep["dropped_outside_year_range"] = int(n1 - len(df))
    rep["rows_final"] = int(len(df))
    rep["year_range"] = (
        [int(df["year"].min()), int(df["year"].max())] if len(df) else None
    )
    self.report["normalization"] = rep
    return df.reset_index(drop=True)

  def cluster(self, df):
    x1, y = self.k_vol, self.k_rate
    df["cluster_label"] = -1
    df["agnostic_profile"] = "Unclassified"
    if y not in df.columns:
      return df
    sub = df[[x1, y]].dropna()
    if len(sub) < 10:
      return df
    feat = sub.copy()
    if self.cl["log_volume"]:
      feat[x1] = np.log10(feat[x1].clip(lower=1))
    Z = StandardScaler().fit_transform(feat)
    rng = np.random.default_rng(self.seed)
    rows, models = [], {}
    kmax = min(self.cl["k_max"], len(sub) - 1)
    for k in range(self.cl["k_min"], kmax + 1):
      km = KMeans(n_clusters=k, n_init=10, random_state=self.seed).fit(Z)
      aris = []
      for b in range(self.cl["stability_runs"]):
        idx = rng.choice(len(Z), len(Z), replace=True)
        kb = KMeans(n_clusters=k, n_init=5, random_state=b).fit(Z[idx])
        aris.append(adjusted_rand_score(km.labels_, kb.predict(Z)))
      rows.append(
          dict(
              k=k,
              silhouette=silhouette_score(
                  Z,
                  km.labels_,
                  sample_size=10000 if len(Z) > 10000 else None,
                  random_state=self.seed,
              ),
              davies_bouldin=davies_bouldin_score(Z, km.labels_),
              wcss=km.inertia_,
              ari_stability_mean=float(np.mean(aris)),
              ari_stability_sd=float(np.std(aris)),
          )
      )
      models[k] = km
    sweep = pd.DataFrame(rows)
    best = int(
        self.cl["fixed_k"] or sweep.loc[sweep["silhouette"].idxmax(), "k"]
    )
    sweep["selected"] = sweep["k"] == best
    sweep.to_csv(
        os.path.join(self.output_dir, "cluster_validation.csv"), index=False
    )
    log.info("K sweep:\n%s", sweep.round(3).to_string(index=False))
    km = models[best]
    df.loc[sub.index, "cluster_label"] = km.labels_
    order = (
        df.loc[sub.index]
        .groupby("cluster_label")[y]
        .mean()
        .sort_values()
        .index.tolist()
    )
    nb = len(self.bands)
    picks = (
        list(range(best))
        if best >= nb
        else sorted(set(np.linspace(0, nb - 1, best).round().astype(int)))
    )
    names = {
        c: (
            self.bands[picks[i]]
            if i < len(picks) and best <= nb
            else (self.bands[i] if i < nb else f"Band {i+1}")
        )
        for i, c in enumerate(order)
    }
    df.loc[sub.index, "agnostic_profile"] = df.loc[
        sub.index, "cluster_label"
    ].map(names)
    cen = km.cluster_centers_
    sep = dict(zip([x1, y], (cen.max(0) - cen.min(0)).round(3).tolist()))
    summ = (
        df[df.cluster_label >= 0]
        .groupby("agnostic_profile")
        .agg(
            n=(x1, "size"),
            volume_median=(x1, "median"),
            rate_mean=(y, "mean"),
            rate_median=(y, "median"),
            rate_p25=(y, lambda s: s.quantile(0.25)),
            rate_p75=(y, lambda s: s.quantile(0.75)),
        )
        .reset_index()
    )
    summ.to_csv(
        os.path.join(self.output_dir, "cluster_summary.csv"), index=False
    )
    self.report["clustering"] = dict(
        k=best,
        centroid_range_std_units=sep,
        unclassified=int((df.cluster_label < 0).sum()),
    )
    if sep.get(y, 0) < 0.5 * sep.get(x1, 0):
      log.warning(
          "Clusters are driven mainly by %s, not %s: bands ranked by mean %s"
          " may be misleading.",
          x1,
          y,
          y,
      )
    return df

  @staticmethod
  def _ols(t, v):
    t, v = np.asarray(t, float), np.asarray(v, float)
    X = sm.add_constant(t)
    m = sm.OLS(v, X).fit()
    hac = sm.OLS(v, X).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
    return dict(
        n_years=int(len(t)),
        slope=float(m.params[1]),
        intercept=float(m.params[0]),
        ci95=[float(x) for x in m.conf_int()[1]],
        p_value=float(m.pvalues[1]),
        p_value_hac=float(hac.pvalues[1]),
        r2=float(m.rsquared),
        durbin_watson=float(durbin_watson(m.resid)),
        resid_var=float(m.scale),
    )

  def trends(self, df):
    x1, out = self.k_vol, {}
    agg = (
        df.groupby("year")[x1]
        .agg(total="sum", per_event="mean", n_events="size")
        .reset_index()
    )
    agg.to_csv(
        os.path.join(self.output_dir, "yearly_aggregates.csv"), index=False
    )
    if len(agg) > 2:
      out["global_total"] = self._ols(agg["year"], agg["total"])
      out["global_per_event_mean"] = self._ols(agg["year"], agg["per_event"])
      out["global_n_events"] = self._ols(agg["year"], agg["n_events"])
      ll = self._ols(agg["year"], np.log(agg["total"].clip(lower=1)))
      out["global_total_loglinear"] = dict(
          ll, annual_growth_pct=float((np.exp(ll["slope"]) - 1) * 100)
      )
    per = {}
    for prof, g in df[df.cluster_label >= 0].groupby("agnostic_profile"):
      a = (
          g.groupby("year")[x1]
          .agg(total="sum", per_event="mean")
          .reset_index()
      )
      if len(a) > 2:
        per[prof] = dict(
            total=self._ols(a["year"], a["total"]),
            per_event_mean=self._ols(a["year"], a["per_event"]),
        )
    out["per_cluster"] = per
    r2 = [v["total"]["r2"] for v in per.values()]
    out["mean_r2_per_cluster_total"] = float(np.mean(r2)) if r2 else None
    out["note"] = (
        "Clusters are defined on the same volume variable that is regressed;"
        " per-cluster slopes are descriptive."
    )
    self.report["trends"] = out
    return agg, out

  def build_views(self, df, trend):
    D = self.display
    d = lambda ident: D.get(ident, ident)
    x1, x2, y, c1, c2 = self.k_vol, self.k_suc, self.k_rate, self.k_cat, self.k_geo
    V = []
    cl = df[df.cluster_label >= 0]
    if y in df.columns and len(cl):
      V.append(
          View(
              "cluster_scatter",
              cl[[x1, y, "agnostic_profile"]].rename(
                  columns={
                      x1: d(x1),
                      y: d(y),
                      "agnostic_profile": "clusterProfile",
                  }
              ),
              f"{self.L['volume']} vs {self.L['rate']} by cluster band",
              "fig1_cluster_scatter.png",
              "Scatter",
              xlabel=d(x1),
              ylabel=d(y),
          )
      )
    if len(cl):
      tp, fits = {}, {}
      for prof, g in cl.groupby("agnostic_profile"):
        tp[prof] = g.groupby("year")[x1].sum()
        if prof in trend.get("per_cluster", {}):
          r = trend["per_cluster"][prof]["total"]
          fits[prof] = (r["slope"], r["intercept"])
      tdf = pd.concat(tp, names=["clusterProfile", "year"]).reset_index(
          name=d(x1)
      )[["year", "clusterProfile", d(x1)]]
      V.append(
          View(
              "trend_by_cluster",
              tdf,
              f"Yearly {self.L['volume']} with per-cluster OLS fits",
              "fig2_cluster_trend_ols.png",
              "Line",
              time="year",
              xlabel="Year",
              ylabel=d(x1),
              fits=fits,
          )
      )
    if c1 in df.columns and df[c1].nunique() > 1:
      a = (
          df.groupby(["year", c1])[x1]
          .sum()
          .reset_index()
          .rename(columns={c1: d(c1), x1: d(x1)})
      )
      V.append(
          View(
              "field_growth",
              a,
              f"Yearly {self.L['volume']} by {self.L['category']}",
              "fig3_field_growth.png",
              "Line",
              time="year",
              stacked=True,
              xlabel="Year",
              ylabel=f"Yearly total {self.L['volume']}",
          )
      )
    if c2 in df.columns:
      g = (
          df.groupby(c2)
          .size()
          .sort_values(ascending=False)
          .head(12)
          .reset_index(name="events")
          .rename(columns={c2: d(c2)})
      )
      V.append(
          View(
              "geo_distribution",
              g,
              f"Distribution by {self.L['location']}",
              "fig4_geo_distribution.png",
              "Geomap",
              geo=d(c2),
              xlabel="Instance count",
              ylabel=d(c2),
          )
      )
    cols = [c for c in ("year", x1, x2, y) if c in df.columns]
    if len(cols) >= 3:
      V.append(
          View(
              "correlation",
              df[cols].rename(columns={x1: d(x1), x2: d(x2), y: d(y)}),
              f"{self.corr_method.capitalize()} correlation of metrics",
              "fig5_correlation.png",
              "Heatmap",
              time="year",
          )
      )
    if c1 in df.columns and y in df.columns:
      sub = df[[c1, y]].dropna().rename(columns={c1: d(c1), y: d(y)})
      V.append(
          View(
              "ratio_by_category",
              sub,
              f"{self.L['rate']} by {self.L['category']}",
              "fig6_ratio_by_category.png",
              "Boxplot",
              xlabel=d(c1),
              ylabel=d(y),
          )
      )
    if "series_acronym" in df.columns:
      t = (
          df.assign(series_acronym=df["series_acronym"].str.upper())
          .groupby("series_acronym")[x1]
          .sum()
          .sort_values(ascending=False)
          .head(10)
          .reset_index()
          .rename(columns={x1: d(x1)})
      )
      V.append(
          View(
              "top_entities",
              t,
              (
                  f"Top {self.L['entity_group']} by cumulative"
                  f" {self.L['volume']}"
              ),
              "fig7_top_entities.png",
              "Bar",
              xlabel=f"Cumulative {self.L['volume']}",
              ylabel=self.L["entity_group"],
          )
      )
    if len(cl):
      V.append(
          View(
              "cluster_density",
              cl[["agnostic_profile", x1]].rename(
                  columns={"agnostic_profile": "clusterProfile", x1: d(x1)}
              ),
              f"{self.L['volume']} distribution by cluster band",
              "fig8_cluster_density.png",
              "Violin",
              xlabel="Cluster Band",
              ylabel=d(x1),
          )
      )
    return V

  @staticmethod
  def _can_render(layout, v):
    num = [
        c
        for c in v.df.columns
        if c != v.time and pd.api.types.is_numeric_dtype(v.df[c])
    ]
    cat = [c for c in v.df.columns if c != v.time and c not in num]
    return {
        "Line": v.time is not None and len(num) >= 1,
        "Bar": bool(cat) and len(num) >= 1,
        "Geomap": v.geo is not None and len(num) >= 1,
        "Scatter": len(num) >= 2,
        "Boxplot": bool(cat) and len(num) >= 1,
        "Violin": bool(cat) and len(num) >= 1,
        "Heatmap": len(num) + (v.time is not None) >= 3,
    }[layout]

  def route_and_render(self, views):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "savefig.dpi": 300,
        "figure.autolayout": True,
    })
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    rows = []
    for v in views:
      shape = describe_view(v.df, v.time, v.geo)
      ranked = self.layout.rank(list(v.df.columns), shape)
      pred = ranked[0][0]
      chosen = next((l for l, _ in ranked if self._can_render(l, v)), None)
      status = (
          "predicted"
          if chosen == pred
          else ("fallback" if chosen else "skipped")
      )
      rows.append(
          dict(
              view=v.name,
              columns=" ".join(v.df.columns),
              shape_tokens=shape,
              predicted=pred,
              confidence=round(ranked[0][1], 3),
              rendered=chosen,
              status=status,
              expert_layout=v.expected,
              agrees_with_expert=(chosen == v.expected),
          )
      )
      if chosen:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        self.renderers[chosen](v, ax)
        ax.set_title(v.title, fontweight="bold", pad=12)
        fig.savefig(os.path.join(self.output_dir, v.fname))
        plt.close(fig)
    rep = pd.DataFrame(rows)
    rep.to_csv(os.path.join(self.output_dir, "routing_report.csv"), index=False)
    self.report["routing"] = dict(
        n_views=len(rep),
        agree_with_expert=int(rep["agrees_with_expert"].sum()),
        classifier_validation=self.layout.validation,
    )
    log.info(
        "Routing:\n%s",
        rep[
            [
                "view",
                "predicted",
                "confidence",
                "rendered",
                "status",
                "agrees_with_expert",
            ]
        ].to_string(index=False),
    )

  # ---- Renderers ----
  @staticmethod
  def _split(v):
    num = [
        c
        for c in v.df.columns
        if c != v.time and pd.api.types.is_numeric_dtype(v.df[c])
    ]
    cat = [c for c in v.df.columns if c != v.time and c not in num]
    return num, cat

  def _r_line(self, v, ax):
    num, cat = self._split(v)
    d = v.df.copy()
    if cat:
      d[cat[0]] = d[cat[0]].map(split_identifier)

    if cat and v.stacked:
      piv = d.pivot_table(
          index=v.time, columns=cat[0], values=num[0], aggfunc="sum"
      ).fillna(0)
      piv.plot(kind="area", stacked=True, ax=ax, cmap="tab20", alpha=0.85)
      ax.legend(title=split_identifier(cat[0]), fontsize=8)
    elif cat:
      for i, (k, g) in enumerate(d.groupby(cat[0])):
        g = g.sort_values(v.time)
        col = sns.color_palette("Set1")[i % 9]
        ax.plot(g[v.time], g[num[0]], marker="o", ms=4, color=col, label=k)
        if k in v.fits:
          s, b = v.fits[k]
          xs = np.array([g[v.time].min(), g[v.time].max()])
          ax.plot(xs, b + s * xs, "--", color=col, lw=1)
      ax.legend(title=split_identifier(cat[0]), fontsize=8)
    else:
      ax.plot(d[v.time], d[num[0]], marker="o")

    if v.time == "year":
      ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel(v.xlabel or split_identifier(v.time))
    ax.set_ylabel(v.ylabel or split_identifier(num[0]))

  def _r_bar(self, v, ax):
    num, cat = self._split(v)
    d = v.df.sort_values(num[0], ascending=False).head(12).copy()
    d[cat[0]] = d[cat[0]].map(split_identifier)
    sns.barplot(data=d, x=num[0], y=cat[0], hue=cat[0], palette="flare", ax=ax)
    if ax.get_legend():
      ax.get_legend().remove()
    ax.set_xlabel(v.xlabel or split_identifier(num[0]))
    ax.set_ylabel(v.ylabel or split_identifier(cat[0]))

  def _r_geo(self, v, ax):
    num, cat = self._split(v)
    d = v.df.sort_values(num[0], ascending=False).head(12).copy()
    d[cat[0]] = d[cat[0]].map(split_identifier)
    sns.barplot(
        data=d,
        x=num[0],
        y=cat[0],
        hue=cat[0],
        palette="Blues_r",
        edgecolor="black",
        ax=ax,
    )
    if ax.get_legend():
      ax.get_legend().remove()
    ax.set_xlabel(v.xlabel or split_identifier(num[0]))
    ax.set_ylabel(v.ylabel or split_identifier(cat[0]))

  def _r_scatter(self, v, ax):
    num, cat = self._split(v)
    d = v.df.copy()
    if cat:
      d[cat[0]] = d[cat[0]].map(split_identifier)
    sns.scatterplot(
        data=d,
        x=num[0],
        y=num[1],
        hue=cat[0] if cat else None,
        palette="Set1",
        s=60,
        alpha=0.8,
        edgecolor="black",
        ax=ax,
    )
    if d[num[0]].max() > 500:
      ax.set_xscale("log")
    ax.set_xlabel(v.xlabel or split_identifier(num[0]))
    ax.set_ylabel(v.ylabel or split_identifier(num[1]))

  def _r_box(self, v, ax):
    num, cat = self._split(v)
    if not num or not cat:
      return
    d = v.df.copy()
    d[cat[0]] = d[cat[0]].map(split_identifier)
    s = d[num[0]].dropna()
    is_ratio = (s.min() >= 0) and (s.max() <= 1.0)

    sns.boxplot(data=d, x=num[0], y=cat[0], hue=cat[0], palette="vlag", ax=ax)
    if ax.get_legend():
      ax.get_legend().remove()

    if is_ratio:
      ax.xaxis.set_major_formatter(
          matplotlib.ticker.PercentFormatter(xmax=1.0)
      )

    ax.set_xlabel(
        v.ylabel
        if v.ylabel and v.ylabel != cat[0]
        else split_identifier(num[0])
    )
    ax.set_ylabel(
        v.xlabel
        if v.xlabel and v.xlabel != num[0]
        else split_identifier(cat[0])
    )

  def _r_violin(self, v, ax):
    num, cat = self._split(v)
    if not num or not cat:
      return
    d = v.df.copy()
    d[cat[0]] = d[cat[0]].map(split_identifier)
    s = d[num[0]].dropna()

    is_ratio = (s.min() >= 0) and (s.max() <= 1.0)
    is_skewed = (s.min() > 0) and (
        s.max() / max(s.min(), 1e-5) > 100 or abs(sps.skew(s)) > 1.5
    )

    if is_skewed and not is_ratio:
      d["_y"] = np.log10(d[num[0]].clip(lower=1e-5))
      ylabel_str = f"{split_identifier(num[0])} (Log10 scale)"
    else:
      d["_y"] = d[num[0]]
      ylabel_str = split_identifier(num[0])

    sns.violinplot(
        data=d, x=cat[0], y="_y", hue=cat[0], palette="Pastel1", cut=0, ax=ax
    )
    if ax.get_legend():
      ax.get_legend().remove()

    if is_ratio and not is_skewed:
      ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))

    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    ax.set_xlabel(
        v.xlabel
        if v.xlabel and v.xlabel != num[0]
        else split_identifier(cat[0])
    )
    ax.set_ylabel(v.ylabel if v.ylabel and v.ylabel != cat[0] else ylabel_str)

  def _r_heat(self, v, ax):
    sns.heatmap(
        v.df.corr(method=self.corr_method, numeric_only=True),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.7,
        ax=ax,
    )

  # ------------------------------------------------------------------ orchestration
  def run_from_dataframe(self, raw_df):
    with self._timer("module1_normalization"):
      df = self.normalize(raw_df)
    if df.empty:
      raise ValueError(
          "No rows left after normalisation (check profile sparql_var bindings"
          " and filters)."
      )
    with self._timer("module3_analytics"):
      df = self.cluster(df)
      yearly, trend = self.trends(df)
    with self._timer("module4_rendering"):
      self.route_and_render(self.build_views(df, trend))
    df.to_csv(os.path.join(self.output_dir, "features_matrix.csv"), index=False)
    self.report["timings_s"] = {
        k: round(v, 3) for k, v in self.timings.items()
    }
    with open(
        os.path.join(self.output_dir, "run_report.json"), "w", encoding="utf-8"
    ) as f:
      json.dump(self.report, f, indent=2, default=str)
    return df

  def execute_pipeline(self, query_str, rdf_url=None, rdf_file=None):
    import rdflib
    from rdflib.util import guess_format

    g = rdflib.Graph()
    with self._timer("module1_ingest_and_sparql"):
      if rdf_url:
        import requests

        r = requests.get(rdf_url, timeout=60)
        r.raise_for_status()
        fmt = guess_format(rdf_url) or (
            "xml" if "xml" in r.headers.get("Content-Type", "") else "turtle"
        )
        g.parse(data=r.content, format=fmt)
      else:
        if not os.path.exists(rdf_file):
          raise FileNotFoundError(rdf_file)
        g.parse(source=rdf_file, format=guess_format(rdf_file) or "turtle")
      self.report["triples"] = len(g)
      res = g.query(query_str)
      cols = [str(v) for v in res.vars]
      rows = [
          [(x.toPython() if x is not None else None) for x in row] for row in res
      ]
      raw = pd.DataFrame(rows, columns=cols)
    if raw.empty:
      raise ValueError("SPARQL projection returned no rows.")
    return self.run_from_dataframe(raw)


def main():
  ap = argparse.ArgumentParser(description="GraphLens v2")
  ap.add_argument(
      "-q", "--query", required=True, help="SPARQL file or https URL"
  )
  ap.add_argument(
      "-p", "--profile", required=True, help="profile.yaml path or https URL"
  )
  ap.add_argument(
      "-t",
      "--training",
      default="data/training_data.csv",
      help="layout training CSV",
  )
  ap.add_argument("-o", "--output", default="output")
  ap.add_argument("--seed", type=int, default=42)
  ap.add_argument("--log-level", default="INFO")
  g = ap.add_mutually_exclusive_group(required=True)
  g.add_argument("-u", "--url")
  g.add_argument("-f", "--file")
  a = ap.parse_args()
  logging.basicConfig(level=a.log_level, format="%(levelname)s %(message)s")
  try:
    if a.query.startswith(("http://", "https://")):
      import requests

      r = requests.get(a.query, timeout=30)
      r.raise_for_status()
      q = r.text
    else:
      with open(a.query, encoding="utf-8") as f:
        q = f.read()
    GraphLensAgnosticPipeline(a.profile, a.training, a.output, a.seed).execute_pipeline(
        q, rdf_url=a.url, rdf_file=a.file
    )
  except Exception as e:
    log.error("%s: %s", type(e).__name__, e)
    sys.exit(1)


if __name__ == "__main__":
  main()