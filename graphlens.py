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
<<<<<<< Updated upstream
import logging
import requests
import argparse
=======
import sys
import time
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
>>>>>>> Stashed changes
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


<<<<<<< Updated upstream
class GraphLensScientometricPipeline:
    def __init__(self, training_data_path="data/training_data.csv", output_dir="output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Core GraphLens Autonomous Layout Classifier Heuristic Setup
        self.layout_clf = DecisionTreeClassifier(criterion="entropy", random_state=42)
        self.vectorizer = CountVectorizer()
        self._bootstrap_layout_classifier(training_data_path)
        
    def _bootstrap_layout_classifier(self, path):
        """Trains the internal heuristic mapping model for layout classification."""
        if os.path.exists(path):
            df = pd.read_csv(path)
            X = df["columns_text"].tolist()
            y = df["target_chart"].tolist()
        else:
            X = [
                "year submitted papers accepted papers", 
                "field acceptanceRate competitiveness", 
                "country total_events",
                "cluster_label volume profile"
            ]
            y = ["Line", "Scatter", "Bar", "ClusterPanel"]
            
        X_vec = self.vectorizer.fit_transform(X)
        self.layout_clf.fit(X_vec, y)
        print("[*] GraphLens ML Layout Engine compiled successfully.")

    def clean_and_normalize_payload(self, df):
        """Performs robust type casting and transforms semantic namespaces into analytical features."""
        if df.empty:
            return df
            
        print("[*] Parsing dates and cleaning semantic URIs into structural features...")
        
        # Extract 4-digit years from the temporal string literals using Regex
        if 'date' in df.columns:
            df['year'] = df['date'].apply(
                lambda x: int(re.search(r'\d{4}', str(x)).group()) 
                if pd.notna(x) and re.search(r'\d{4}', str(x)) else np.nan
            )
        else:
            df['year'] = np.nan

        # Strip remaining namespace URIs to cleanly isolate literal tokens for charting labels
        for col in df.columns:
            if col not in ['date', 'year']:
                df[col] = df[col].apply(
                    lambda x: str(x).strip().split('/')[-1].split('#')[-1] if pd.notna(x) else np.nan
                )
        
        # Explicitly cast metric arrays to float numerical arrays
        numeric_targets = ['submitted', 'accepted', 'acceptanceRate']
        for col in numeric_targets:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Cross-derive missing metrics mathematically to maximize valid observational sample sizes
        if 'submitted' in df.columns:
            if 'accepted' in df.columns and 'acceptanceRate' in df.columns:
                mask_acc = df['accepted'].isna() & df['acceptanceRate'].notna() & (df['submitted'] > 0)
                df.loc[mask_acc, 'accepted'] = (df.loc[mask_acc, 'submitted'] * df.loc[mask_acc, 'acceptanceRate']).round()
                
                mask_rate = df['acceptanceRate'].isna() & (df['submitted'] > 0) & df['accepted'].notna()
                df.loc[mask_rate, 'acceptanceRate'] = df.loc[mask_rate, 'accepted'] / df.loc[mask_rate, 'submitted']
            
            # Normalize percentage variants to a standard [0.0, 1.0] decimal format
            if 'acceptanceRate' in df.columns and df['acceptanceRate'].max() > 1.0:
                df['acceptanceRate'] = df['acceptanceRate'] / 100.0
                
        # Drop entries missing critical timeline vectors
        df = df.dropna(subset=['year', 'submitted']).copy()
        df['year'] = df['year'].astype(int)
        
        return df

    def execute_advanced_analytics(self, df):
        """Applies unsupervised learning and statistical regression over the processed data matrix."""
        print("[*] Running advanced Scientometric discovery engine...")
        
        # 1. Unsupervised Machine Learning: K-Means Clustering on Competitive Behavior
        clustering_data = df[['submitted', 'acceptanceRate']].dropna()
        if len(clustering_data) >= 3:
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            labels = kmeans.fit_predict(clustering_data)
            df.loc[clustering_data.index, 'cluster_label'] = labels
            
            # Categorize numeric cluster coordinates into standardized academic taxa profiles
            cluster_means = df.groupby('cluster_label')['acceptanceRate'].mean()
            sorted_clusters = cluster_means.sort_values().index
            cluster_mapping = {
                sorted_clusters[0]: "Elite / Highly Selective",
                sorted_clusters[1]: "Standard Peer-Reviewed",
                sorted_clusters[2]: "High-Acceptance / Mega-Venue"
            }
            df['venue_profile'] = df['cluster_label'].map(cluster_mapping).fillna("Unclassified Venues")
        else:
            df['venue_profile'] = "Unclassified Venues"

        # 2. Statistical Modeling: Ordinary Least Squares (OLS) Trend Line Regression
        yearly_aggregates = df.groupby('year').agg({'submitted': 'sum'}).reset_index()
        if len(yearly_aggregates) > 1:
            X_reg = yearly_aggregates[['year']].values
            y_reg = yearly_aggregates['submitted'].values
            reg_model = LinearRegression().fit(X_reg, y_reg)
            df_meta = {
                "growth_slope": reg_model.coef_[0],
                "r_squared": reg_model.score(X_reg, y_reg)
            }
            print(f"[+] Empirical growth trend computed: {df_meta['growth_slope']:.2f} submissions/year (R² = {df_meta['r_squared']:.2f})")
        else:
            df_meta = {"growth_slope": 0.0, "r_squared": 0.0}
=======
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
>>>>>>> Stashed changes

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

<<<<<<< Updated upstream
    def generate_publication_plots(self, df, yearly_df, meta):
        """Generates and exports high-resolution multi-panel plots optimized for journal formatting."""
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
        plt.rcParams.update({
            'font.family': 'serif',
            'savefig.dpi': 300,
            'figure.autolayout': True
        })

        print("\n[*] Generating high-resolution graphical assets and terminal captions...\n" + "="*80)

        # FIGURE 1: Cluster Analysis with Overlaid OLS Trend Fit
        if 'venue_profile' in df.columns and len(df['venue_profile'].unique()) > 1:
            fig1, ax1 = plt.subplots(figsize=(9, 6))
            sns.scatterplot(
                data=df, x='submitted', y='acceptanceRate', hue='venue_profile',
                palette='Set1', s=110, alpha=0.8, edgecolor='black', ax=ax1
            )
            sns.regplot(
                data=df, x='submitted', y='acceptanceRate', scatter=False,
                color='gray', line_kws={'linestyle': '--', 'linewidth': 1.5}, ax=ax1
            )
            ax1.set_title("Figure 1: K-Means Clustering and OLS Trend Matrix of Scholarly Venues", fontweight='bold', pad=12)
            ax1.set_xlabel("Submission Ingestion Volume (Log Scale)", fontsize=12)
            ax1.set_ylabel("Empirical Review Acceptance Rate", fontsize=12)
            ax1.set_xscale('log')
            ax1.legend(title="Structural Taxonomy", frameon=True, loc='best')
            
            fig1_path = os.path.join(self.output_dir, "fig1_cluster_regression.png")
            plt.savefig(fig1_path, dpi=300)
            plt.close(fig1)
            print(f"[+] Exported: {fig1_path}")
            print("    CAPTION: Figure 1: Multi-dimensional review behavior taxonomy mapping venue intake volumes against metric acceptance rates. Points categorize automatically via unsupervised K-Means clusters (K=3), overlaid with an OLS trend curve modeling structural competitiveness scaling patterns.\n" + "-"*80)

        # FIGURE 2: Longitudinal Distribution of Submissions Across Fields
        if 'field' in df.columns and len(df['field'].dropna().unique()) > 1:
            field_temporal = df.groupby(['year', 'field'])['submitted'].sum().unstack().fillna(0)
            
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            field_temporal.plot(kind='area', stacked=True, cmap='tab20', alpha=0.85, ax=ax2)
            ax2.set_title("Figure 2: Longitudinal Evolution of Submissions Across Specialization Fields", fontweight='bold', pad=12)
            ax2.set_xlabel("Academic Tracking Year", fontsize=12)
            ax2.set_ylabel("Cumulative Volume Count", fontsize=12)
            ax2.legend(title="Research Field", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
            
            fig2_path = os.path.join(self.output_dir, "fig2_field_temporal_growth.png")
            plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
            plt.close(fig2)
            print(f"[+] Exported: {fig2_path}")
            print("    CAPTION: Figure 2: Stacked longitudinal area matrix tracing total thematic distributions across computer science communities across five decades, exposing macro-scale structural shifts and field specialization pathways within knowledge networks.\n" + "-"*80)

        # FIGURE 3: Geographic Distribution of Conference Hosting Nations
        if 'country' in df.columns and len(df['country'].dropna().unique()) > 1:
            geo_dist = df.groupby('country')['event'].nunique().sort_values(ascending=False).head(12)
            
            fig3, ax3 = plt.subplots(figsize=(9, 5.5))
            sns.barplot(x=geo_dist.values, y=geo_dist.index, palette='Blues_r', edgecolor='black', ax=ax3)
            ax3.set_title("Figure 3: Geographic Concentration Profile of Renowned Event Frameworks", fontweight='bold', pad=12)
            ax3.set_xlabel("Absolute Count of Unique Evaluated Conference Editions", fontsize=12)
            ax3.set_ylabel("Hosting Nation Entity", fontsize=12)
            
            fig3_path = os.path.join(self.output_dir, "fig3_geographic_distribution.png")
            plt.savefig(fig3_path, dpi=300)
            plt.close(fig3)
            print(f"[+] Exported: {fig3_path}")
            print("    CAPTION: Figure 3: Concentration distribution profiling geographic host asymmetry of international scientific tracking instances, ranking top nations based on their cumulative count of elite conference events.\n" + "-"*80)

        # FIGURE 4: Metrics Dependency Matrix Correlation Heatmap
        target_metrics = [c for c in ['year', 'submitted', 'accepted', 'acceptanceRate'] if c in df.columns]
        if len(target_metrics) > 1:
            fig4, ax4 = plt.subplots(figsize=(7.5, 6))
            corr_matrix = df[target_metrics].corr(method='pearson')
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.7, cbar=True, ax=ax4)
            ax4.set_title("Figure 4: Correlation Feature Space Heatmap of Core Scholarly Metrics", fontweight='bold', pad=12)
            
            fig4_path = os.path.join(self.output_dir, "fig4_correlation_matrix.png")
            plt.savefig(fig4_path, dpi=300)
            plt.close(fig4)
            print(f"[+] Exported: {fig4_path}")
            print("    CAPTION: Figure 4: Correlation heatmap illustrating Pearson linear dependency indices between temporal progression, submission volume scales, absolute acceptance counts, and review selectiveness constraints.\n" + "-"*80)

        # FIGURE 5: Operational Selectivity Variances Across Fields (Boxplot)
        if 'field' in df.columns and 'acceptanceRate' in df.columns:
            valid_fields = df.dropna(subset=['field', 'acceptanceRate'])
            if len(valid_fields['field'].unique()) > 1:
                fig5, ax5 = plt.subplots(figsize=(10, 6))
                sns.boxplot(data=valid_fields, x='acceptanceRate', y='field', palette='vlag', fliersize=4, ax=ax5)
                ax5.set_title("Figure 5: Variance Dynamics of Review Selectiveness Across Fields", fontweight='bold', pad=12)
                ax5.set_xlabel("Empirical Acceptance Rate (Decimal Range)", fontsize=12)
                ax5.set_ylabel("Research Field Taxonomy", fontsize=12)
                
                fig5_path = os.path.join(self.output_dir, "fig5_field_selectivity_boxplot.png")
                plt.savefig(fig5_path, dpi=300)
                plt.close(fig5)
                print(f"[+] Exported: {fig5_path}")
                print("    CAPTION: Figure 5: Cross-domain selectivity variance analysis utilizing horizontal boxplots. This visualization illustrates the structural variation, inner quartiles, and outlier thresholds of peer-review stringency across distinct academic communities.\n" + "-"*80)

        # FIGURE 6: Top 10 Conference Series Benchmarking (Bar Chart)
        if 'acronym' in df.columns and 'submitted' in df.columns:
            top_venues = df.groupby('acronym')['submitted'].sum().sort_values(ascending=False).head(10)
            if not top_venues.empty:
                fig6, ax6 = plt.subplots(figsize=(10, 6))
                sns.barplot(x=top_venues.values, y=top_venues.index, palette='flare', edgecolor='black', ax=ax6)
                ax6.set_title("Figure 6: Benchmark Profile of Top 10 High-Impact Venues by Volume", fontweight='bold', pad=12)
                ax6.set_xlabel("Cumulative Tracked Submissions Ingested", fontsize=12)
                ax6.set_ylabel("Venue Acronym Identifier", fontsize=12)
                
                fig6_path = os.path.join(self.output_dir, "fig6_top_venues_benchmark.png")
                plt.savefig(fig6_path, dpi=300)
                plt.close(fig6)
                print(f"[+] Exported: {fig6_path}")
                print("    CAPTION: Figure 6: Evaluation benchmark ranking the top ten most active publication series by historical submission volume within the triplestore metadata repository, highlighting dominant nodes of research productivity.\n" + "-"*80)

        # FIGURE 7: Density Volume Distribution by Taxonomy Cluster (Violin Plot)
        if 'venue_profile' in df.columns and 'submitted' in df.columns:
            fig7, ax7 = plt.subplots(figsize=(10, 6))
            sns.violinplot(data=df, x='venue_profile', y='submitted', palette='Pastel1', inner='quartile', ax=ax7)
            ax7.set_yscale('log')
            ax7.set_title("Figure 7: Density Distribution of Submission Volumes by Taxonomical Cluster", fontweight='bold', pad=12)
            ax7.set_xlabel("Unsupervised Taxonomy Profile Assignment", fontsize=12)
            ax7.set_ylabel("Inbound Submission Scale (Logarithmic Scale)", fontsize=12)
            
            fig7_path = os.path.join(self.output_dir, "fig7_cluster_density_violin.png")
            plt.savefig(fig7_path, dpi=300)
            plt.close(fig7)
            print(f"[+] Exported: {fig7_path}")
            print("    CAPTION: Figure 7: Log-scaled violin plots tracking kernel density distributions of submission volumes across unsupervised venue profiles. This map exposes structural density thresholds and scale differences among elite, standard, and mega-venues.\n" + "="*80)

    def execute_pipeline(self, query_str, rdf_url=None, rdf_file=None):
        """Ingests semantic data from remote URLs or local files with explicit layout structure validation."""
        graph = rdflib.Graph()
        detected_format = None

        if rdf_url:
            print(f"[*] Connecting to live semantic data channel: {rdf_url}")
            try:
                response = requests.get(rdf_url, timeout=30)
                if response.status_code == 404:
                    print(f"\n[!] Fatal Ingestion Error: The requested URL returned a '404 Not Found' status.")
                    return
                response.raise_for_status()
                
                detected_format = guess_format(rdf_url)
                if not detected_format:
                    content_type = response.headers.get('Content-Type', '').lower()
                    if 'turtle' in content_type or 'ttl' in content_type: detected_format = 'turtle'
                    elif 'rdf+xml' in content_type or 'xml' in content_type: detected_format = 'xml'
                    elif 'json-ld' in content_type: detected_format = 'json-ld'
                    elif 'ntriples' in content_type: detected_format = 'nt'
                
                if not detected_format:
                    print("\n[!] Fatal Ingestion Error: Could not determine RDF serialization framework from URL metadata strings.")
                    return

                try:
                    graph.parse(data=response.text, format=detected_format)
                except Exception as parse_error:
                    print(f"\n[!] Fatal Structural Error: Target stream is not a valid RDF payload format ({detected_format}).\n    Parser Details: {parse_error}")
                    return

            except requests.exceptions.RequestException as network_err:
                print(f"\n[!] Ingestion Failure: Unable to securely connect to remote service channel.\n    Network Details: {network_err}")
                return

        elif rdf_file:
            print(f"[*] Loading local semantic data asset: {rdf_file}")
            if not os.path.exists(rdf_file):
                print(f"\n[!] Fatal File Error: The configured system storage path '{rdf_file}' does not exist.")
                return
            
            detected_format = guess_format(rdf_file)
            if not detected_format:
                print("\n[!] Fatal Configuration Error: Extension schema from target file name matches no known RDF serialization format layout.")
                return
                
            try:
                graph.parse(source=rdf_file, format=detected_format)
            except Exception as parse_error:
                print(f"\n[!] Fatal Structural Error: Provided file context is not a valid RDF serialization format layout ({detected_format}).\n    Parser Details: {parse_error}")
                return
        else:
            print("\n[!] Technical Ingestion Bound: No active input stream parameter path provided.")
            return

        print(f"[+] Triplestore parsed successfully into memory using layout parsing mode: '{detected_format}'. Size: {len(graph)} triples.")
        print("[*] Compiling schema extraction query parsing operations...")
        
        try:
            query_job = graph.query(query_str)
        except Exception as query_error:
            print(f"\n[!] Fatal SPARQL Syntax Error: The external query file could not be compiled.\n    Details: {query_error}")
            return

        columns = [str(var) for var in query_job.vars]
        
        rows = []
        for row in query_job:
            rows.append([val.toPython() if hasattr(val, 'toPython') else str(val) if val is not None else None for val in row])
            
        raw_df = pd.DataFrame(rows, columns=columns)
        if raw_df.empty:
            print("[!] Fatal Error: Extraction query generated an empty dataframe context. Verify your query's URI namespaces.")
            return

        print(f"[+] Raw payload captured successfully: {len(raw_df)} metadata rows loaded.")

        # Execute analytical processing sub-modules
        processed_df = self.clean_and_normalize_payload(raw_df)
        analyzed_df, yearly_df, metadata = self.execute_advanced_analytics(processed_df)
        self.generate_publication_plots(analyzed_df, yearly_df, metadata)
        
        csv_path = os.path.join(self.output_dir, "processed_scientometric_matrix.csv")
        analyzed_df.to_csv(csv_path, index=False)
        print(f"\n[+++] Success! All 7 visual assets and structural tables saved to: '{self.output_dir}/'\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphLens: Automated Semantic Graph Scientometric Pipeline")
    
    # Query configuration file input parameter
    parser.add_argument("-q", "--query", required=True, help="Path to external SPARQL query file containing prefix definitions (.rq, .sparql)")
    
    # Input data channels
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-u", "--url", help="Remote url target channel pointing to an RDF graph serialization (.ttl, .rdf, .jsonld)")
    group.add_argument("-f", "--file", help="Local storage file path pointing directly to an RDF serialization data asset")
    
    args = parser.parse_args()
    
    # Read query file context safely
    if not os.path.exists(args.query):
        print(f"[!] Fatal Configuration Error: The specified query file '{args.query}' was not found.")
        exit(1)
        
    with open(args.query, "r", encoding="utf-8") as f:
        loaded_sparql_query = f.read()
    
    pipeline = GraphLensScientometricPipeline()
    pipeline.execute_pipeline(
        query_str=loaded_sparql_query, 
        rdf_url=args.url, 
        rdf_file=args.file
    )
=======
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
>>>>>>> Stashed changes
