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
import textwrap
import sys
import time
import unicodedata
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
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
DEFAULT_YEAR_PATTERN = r"(?<!\d)(\d{4})(?!\d)"  # any 4-digit year; bound it with filters.year_min/max
RATIO_SCALES = ("auto", "fraction", "percent", "unbounded")
GEO_SHAPES_DEFAULT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "world_countries.geojson"
)
# Common spellings that do not appear in the Natural Earth name fields -> ADM0_A3 code.
GEO_ALIASES = {
    "usa": "USA", "us": "USA", "unitedstates": "USA", "unitedstatesofamerica": "USA",
    "uk": "GBR", "greatbritain": "GBR", "britain": "GBR", "england": "GBR",
    "scotland": "GBR", "wales": "GBR", "northernireland": "GBR",
    "southkorea": "KOR", "korea": "KOR", "republicofkorea": "KOR",
    "russia": "RUS", "russianfederation": "RUS",
    "czechrepublic": "CZE", "czechia": "CZE",
    "macedonia": "MKD", "northmacedonia": "MKD", "republicofmacedonia": "MKD",
    "turkey": "TUR", "turkiye": "TUR", "uae": "ARE", "holland": "NLD",
    "ivorycoast": "CIV", "vietnam": "VNM", "hongkong": "HKG", "macau": "MAC",
    "taiwan": "TWN", "chinesetaipei": "TWN", "westgermany": "DEU", "eastgermany": "DEU",
    "congokinshasa": "COD", "drcongo": "COD", "drc": "COD", "congobrazzaville": "COG",
    # earlier names of the same state (unambiguous successors only)
    "weimarrepublic": "DEU", "nazigermany": "DEU", "germanempire": "DEU",
    "kingdomoffrance": "FRA", "kingdomofitaly": "ITA", "polishpeoplesrepublic": "POL",
}


# --------------------------------------------------------------------------- helpers
def shannon_entropy(series, normalize=True):
  """Shannon entropy H(X) (Eq. 1) of a column's empirical value distribution."""
  p = series.dropna().value_counts(normalize=True).values
  p = p[p > 0]   # unobserved categories (e.g. empty bins) carry no information
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


def _compact(x, _pos=None):
  """Axis tick text for large numbers: 250k, 3.5M, 1.2B (instead of a '1e8' offset)."""
  for div, unit in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
    if abs(x) >= div:
      return f"{x / div:,.3g}{unit}"
  return f"{x:,.0f}" if float(x).is_integer() else f"{x:g}"


def _short(text, n=32):
  """Display form of a long category label (data values are never changed)."""
  t = str(text)
  return t if len(t) <= n else t[: n - 1].rstrip() + "\u2026"


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


def _geo_key(name):
  """'United_States' / 'Georgia (country)' / 'Türkiye' / 'Cinema of Ireland'
  -> 'unitedstates' / 'georgia' / 'turkiye' / 'ireland'."""
  s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
  s = re.sub(r"\(.*?\)", "", s).replace("_", " ").strip().lower()
  s = re.sub(r"^(cinema|film industry) (of|in) ", "", s)   # DBpedia links films to 'Cinema_of_X'
  s = re.sub(r"^the\s+", "", s)
  return re.sub(r"[^a-z0-9]", "", s)


def _geo_lookup(world, name):
  """Country code for a location name. A bracketed qualifier is tried first, so
  'Congo (Kinshasa)' is not reduced to the plain 'Congo' (another country)."""
  if world is None or name is None or (isinstance(name, float) and np.isnan(name)):
    return None
  s = str(name)
  if "(" in s:
    hit = world["lookup"].get(_geo_key(s.replace("(", " ").replace(")", " ")))
    if hit:
      return hit
  return world["lookup"].get(_geo_key(s))


def load_world_shapes(path):
  """Country outlines from a GeoJSON file -> {'paths', 'lookup', 'names', 'anchor', 'area'}; None if unavailable."""
  if not path or not os.path.exists(path):
    log.warning("Geomap basemap not found at %s -> falling back to a bar chart.", path)
    return None
  with open(path, encoding="utf-8") as f:
    feats = json.load(f).get("features", [])
  W = dict(paths={}, lookup={}, names={}, anchor={}, area={})
  name_fields = ("NAME", "NAME_LONG", "ADMIN", "NAME_EN", "FORMAL_EN", "NAME_SORT", "NAME_ALT", "ABBREV")
  for ft in feats:
    p, g = ft.get("properties", {}), ft.get("geometry") or {}
    cid = p.get("ADM0_A3") or p.get("ISO_A3") or p.get("NAME")
    polys = (
        [g["coordinates"]] if g.get("type") == "Polygon"
        else g.get("coordinates", []) if g.get("type") == "MultiPolygon" else []
    )
    rings = [np.asarray(r, float) for poly in polys for r in poly if len(r) >= 3]
    if not rings:
      continue
    W["paths"][cid] = MplPath.make_compound_path(*[MplPath(r, closed=True) for r in rings])
    ext = [np.asarray(poly[0], float) for poly in polys if len(poly[0]) >= 3]
    areas = [0.5 * abs(np.dot(r[:, 0], np.roll(r[:, 1], 1)) - np.dot(r[:, 1], np.roll(r[:, 0], 1))) for r in ext]
    W["area"][cid] = float(sum(areas))
    W["anchor"][cid] = ext[int(np.argmax(areas))].mean(axis=0)
    W["names"][cid] = p.get("NAME", cid)
    for k in name_fields:
      if p.get(k):
        W["lookup"].setdefault(_geo_key(p[k]), cid)
    for k in ("ISO_A3", "ADM0_A3", "ISO_A2"):
      if p.get(k):
        W["lookup"].setdefault(_geo_key(p[k]), cid)
  for alias, cid in GEO_ALIASES.items():
    if cid in W["paths"]:
      W["lookup"][alias] = cid
  return W


# --------------------------------------------------------------------------- SPARQL endpoint
_PROLOGUE_RE = re.compile(
    r"^((?:\s*(?:#[^\n]*|PREFIX\s+[^\s:]*:\s*<[^>]*>|BASE\s*<[^>]*>))*)(.*)$",
    re.IGNORECASE | re.DOTALL,
)
_HAS_LIMIT_RE = re.compile(r"\b(LIMIT|OFFSET)\s+\d+\s*$", re.IGNORECASE)
# '# GRAPHLENS-PARTITION ?var' in a query marks where a hash-slice FILTER is inserted
_PARTITION_RE = re.compile(r"#[ \t]*GRAPHLENS-PARTITION[ \t]+\?(\w+)[^\n]*", re.IGNORECASE)


def _strip_comments(q):
  """Remove # comments that are not inside IRIs or string literals."""
  out, i, n = [], 0, len(q)
  while i < n:
    c = q[i]
    if c in "\"'":
      j = i + 1
      while j < n and q[j] != c:
        j += 2 if q[j] == "\\" else 1
      out.append(q[i:j + 1]); i = j + 1
    elif c == "<":
      j = q.find(">", i)
      if j == -1 or any(ch.isspace() for ch in q[i:j]):
        out.append(c); i += 1          # a less-than operator, not an IRI
      else:
        out.append(q[i:j + 1]); i = j + 1
    elif c == "#":
      j = q.find("\n", i)
      i = n if j == -1 else j
    else:
      out.append(c); i += 1
  return "".join(out)


class TransientEndpointError(RuntimeError):
  """A request kept failing for reasons that may go away (network, overload, timeouts)."""


class SparqlEndpoint:
  """Minimal SPARQL 1.1 Protocol client with LIMIT/OFFSET paging and retries.

  Bindings are converted with rdflib (URIRef / Literal.toPython), so values are typed
  exactly as in file mode (xsd:integer -> int, xsd:dateTime -> datetime, ...).
  """

  RETRY_STATUS = (429, 500, 502, 503, 504)

  def __init__(self, url, page_size=5000, max_rows=None, timeout=120, retries=4,
               stable_order=False, count_check=True, min_page_size=250, backoff=2.0,
               partitions=16,
               user_agent="GraphLens/2.0 (https://github.com/saidfathalla/graphlens)"):
    if page_size < 1:
      raise ValueError("page_size must be >= 1")
    self.url, self.page_size, self.max_rows = url, int(page_size), max_rows
    self.timeout, self.retries, self.stable_order = timeout, retries, stable_order
    self._last_partial = False
    self.count_check = count_check
    self.min_page_size = max(1, min(int(min_page_size), self.page_size))
    self.backoff = float(backoff)
    if partitions not in (1, 16, 256):
      raise ValueError("partitions must be 1, 16 or 256")
    self.partitions = partitions
    self.headers = {"Accept": "application/sparql-results+json", "User-Agent": user_agent}
    self.stats = dict(endpoint=url, pages=0, requests=0, rows=0, paged=True,
                      stable_order=stable_order, expected_rows=None, complete=None,
                      incomplete_responses=0, duplicate_rows=0, failed_requests=0,
                      page_size_reduced_to=None, partitions=1, expected_is_lower_bound=False,
                      sample_partitions=None)

  def _post(self, query, retries=None, allow_partial=True):
    """POST a query; retry transient failures (dropped connections, truncated bodies,
    HTTP 429/5xx) with exponential back-off, then raise TransientEndpointError."""
    import requests
    transient = (requests.ConnectionError, requests.Timeout,
                 requests.exceptions.ChunkedEncodingError, requests.exceptions.ContentDecodingError)
    retries = self.retries if retries is None else retries
    delay = self.backoff
    for attempt in range(retries + 1):
      self.stats["requests"] += 1
      try:
        r = requests.post(self.url, data={"query": query}, headers=self.headers, timeout=self.timeout)
      except transient as e:
        err = f"{type(e).__name__}: {str(e)[:160]}"
      else:
        if r.status_code in self.RETRY_STATUS:
          err = f"HTTP {r.status_code}"
          ra = r.headers.get("Retry-After", "")
          delay = float(ra) if ra.isdigit() else delay
        elif r.status_code >= 400:
          raise RuntimeError(f"SPARQL endpoint returned HTTP {r.status_code}: {r.text[:300]}")
        else:
          try:
            data = r.json()
          except ValueError as e:              # body cut off or not JSON
            err = f"invalid or truncated JSON ({str(e)[:80]})"
          else:
            state = r.headers.get("X-SQL-State", "") + " " + r.headers.get("X-SQL-Message", "")
            self._last_partial = "S1TAT" in state or "incomplete" in state.lower()
            if self._last_partial:
              self.stats["incomplete_responses"] += 1
            if self._last_partial and not allow_partial:
              err = "partial result: the endpoint hit its time limit"
            else:
              if self._last_partial:
                log.warning("Endpoint returned a PARTIAL result (query hit its time limit)")
              return data
      self.stats["failed_requests"] += 1
      if attempt == retries:
        raise TransientEndpointError(f"SPARQL request failed after {attempt + 1} attempt(s): {err}")
      log.warning("SPARQL request failed (%s); attempt %d/%d, retrying in %.0fs",
                  err, attempt + 1, retries + 1, delay)
      time.sleep(delay)
      delay = min(delay * 2, 60)

  @staticmethod
  def _term(b):
    from rdflib import BNode, Literal, URIRef
    if b is None:
      return None
    t, v = b.get("type"), b.get("value")
    if t == "uri":
      return URIRef(v).toPython()
    if t == "bnode":
      return BNode(v).toPython()
    dt = b.get("datatype")
    return Literal(v, datatype=URIRef(dt) if dt else None, lang=b.get("xml:lang")).toPython()

  @staticmethod
  def _join(frames, cols):
    """Join pages/partitions so column types are inferred once, as for a single result
    (a page whose column is entirely empty would otherwise force a generic type)."""
    return pd.DataFrame([r for f in frames for r in f.itertuples(index=False, name=None)],
                        columns=cols)

  def _frame(self, res, cols=None):
    cols = cols or res["head"]["vars"]
    rows = [[self._term(b.get(c)) for c in cols] for b in res["results"]["bindings"]]
    return pd.DataFrame(rows, columns=cols)

  def select(self, query):
    """Run a SELECT query, transparently paging through the full result.

    If the query contains '# GRAPHLENS-PARTITION ?var', it is run as 16 (or 256)
    independent slices, FILTER(SUBSTR(MD5(STR(?var)), 1, k) = "<hex prefix>"), so that
    aggregating queries stay small enough for public endpoints' time limits.
    With max_rows, whole slices are fetched until max_rows is reached: since slices are
    hash buckets, the result is a reproducible random sample of the entities.
    """
    m = _PARTITION_RE.search(query)
    self._so_far, self._slice = 0, ""          # progress shown in the log
    if m and self.partitions > 1:
      var, width = m.group(1), 1 if self.partitions == 16 else 2
      prefixes = [format(i, f"0{width}x") for i in range(16 ** width)]
      self.stats.update(partitions=len(prefixes), partition_var=var)
      frames, expected, complete = [], 0, True
      limit, self.max_rows = self.max_rows, None     # slices are always fetched whole
      try:
        for i, pre in enumerate(prefixes, 1):
          qi = (query[: m.start()] + f'FILTER(SUBSTR(MD5(STR(?{var})), 1, {width}) = "{pre}")'
                + query[m.end():])
          self._slice = f"Slice {i}/{len(prefixes)}, "
          df_i, exp_i, ok_i = self._select_one(qi)
          if df_i.empty:
            log.info("Slice %d/%d: 0 rows | retrieved so far: %s", i, len(prefixes),
                     f"{self._so_far:,}")
          frames.append(df_i)
          expected = None if exp_i is None or expected is None else expected + exp_i
          complete = None if ok_i is None or complete is None else complete and ok_i
          if limit is not None and sum(map(len, frames)) >= limit and i < len(prefixes):
            self.stats.update(sample_partitions=i)
            log.info("Sample: stopping after %d of %d slices (%.0f%% of ?%s values, chosen by "
                     "hash) because --max-rows %s was reached", i, len(prefixes),
                     100 * i / len(prefixes), var, f"{limit:,}")
            break
      finally:
        self.max_rows = limit
      df = self._join(frames, list(frames[0].columns))
    else:
      df, expected, complete = self._select_one(query)
    dups = int(df.astype(str).duplicated().sum()) if len(df) else 0
    if dups:
      complete = False
      log.warning("%d identical rows arrived on different pages: the endpoint changed row "
                  "order between requests, so other rows may be missing. Use --stable-order "
                  "or a smaller result.", dups)
    self.stats.update(rows=len(df), expected_rows=expected, complete=complete, duplicate_rows=dups)
    if complete is False and expected is not None and len(df) < expected:
      log.warning("INCOMPLETE RESULT overall: %d of %d%s rows (%.0f%%).", len(df),
                  expected, "+" if self.stats["expected_is_lower_bound"] else "",
                  100 * len(df) / max(expected, 1))
    return df

  def _select_one(self, query):
    """Page through one query. Returns (frame, expected_rows or None, complete or None)."""
    q = _strip_comments(query).strip()
    if not re.search(r"\bSELECT\b", q, re.IGNORECASE):
      raise ValueError("Endpoint mode supports SELECT queries only.")
    if _HAS_LIMIT_RE.search(q):            # the query pages itself: run it once
      self.stats.update(paged=False)
      self.stats["pages"] += 1
      return self._frame(self._post(q, allow_partial=False)), None, None
    prologue, body = _PROLOGUE_RE.match(q).groups()
    if self.stable_order:                  # deterministic pages: ORDER BY all projected vars
      head = self._post(f"{prologue}\n{body}\nLIMIT 1")["head"]["vars"]   # original projection order
      proj = " ".join(f"?{v}" for v in head)
      base = f"{prologue}\nSELECT {proj} WHERE {{\n{body}\n}}\nORDER BY {proj}\n"
    else:
      base = f"{prologue}\n{body}\n"
    frames, offset, cols, npage = [], 0, None, 0
    while True:
      n = self.page_size if self.max_rows is None else min(self.page_size, self.max_rows - offset)
      if n <= 0:
        break
      try:   # large pages get one retry; if that fails too, the page size is halved
        res = self._post(f"{base}LIMIT {n} OFFSET {offset}",
                         retries=1 if n > self.min_page_size else None, allow_partial=False)
      except TransientEndpointError as e:
        if n <= self.min_page_size:
          hint = (" This query aggregates (GROUP BY), so every page re-runs it in full: add "
                  "'# GRAPHLENS-PARTITION ?<entity variable>' inside its WHERE clause to split "
                  "it into small slices, or use a local copy of the data."
                  if re.search(r"\bGROUP\s+BY\b", body, re.IGNORECASE) and
                  self.stats["partitions"] == 1 else " Try again later or use a local copy.")
          raise TransientEndpointError(f"{e} (page of {n} rows at offset {offset}).{hint}") from e
        self.page_size = max(self.min_page_size, n // 2)
        self.stats["page_size_reduced_to"] = self.page_size
        log.warning("Page of %d rows kept failing; continuing with %d rows per page",
                    n, self.page_size)
        continue
      cols = cols or res["head"]["vars"]
      page = self._frame(res, cols)
      if page.empty:                       # a complete, empty page is the only reliable end
        break                              # marker, because endpoints cap rows per request
      frames.append(page)
      self.stats["pages"] += 1
      offset += len(page)
      npage += 1
      self._so_far = getattr(self, "_so_far", 0) + len(page)
      log.info("%s%s %d: %s rows | retrieved so far: %s", getattr(self, "_slice", ""),
               "page" if getattr(self, "_slice", "") else "SPARQL page", npage,
               f"{len(page):,}", f"{self._so_far:,}")
    df = self._join(frames, cols or [])
    expected, complete = self._count(prologue, body, len(df))
    return df, expected, complete

  def _count(self, prologue, body, got):
    """Ask the endpoint how many rows the query has; compare with what paging returned."""
    if not self.count_check:
      return None, None
    try:
      res = self._post(f"{prologue}\nSELECT (COUNT(*) AS ?n) WHERE {{\n{body}\n}}")
      expected = int(res["results"]["bindings"][0]["n"]["value"])
    except Exception as e:                  # counting is best effort; never fail the run
      log.info("Could not verify result size with COUNT (%s)", e)
      return None, None
    lower = self._last_partial              # a timed-out COUNT is only a lower bound
    if lower:
      self.stats["expected_is_lower_bound"] = True
    target = expected if self.max_rows is None else min(expected, self.max_rows)
    if got < target:
      log.warning(
          "INCOMPLETE RESULT: the endpoint reports %s%d rows but paging returned %d. %s",
          "at least " if lower else "", expected, got,
          "The query uses ORDER BY, which Virtuoso endpoints such as DBpedia cannot page "
          "deeply; remove it." if re.search(r"\bORDER\s+BY\b", body, re.IGNORECASE) else
          "Add '# GRAPHLENS-PARTITION ?<entity variable>' to split the query, narrow it "
          "with FILTERs, or use a local copy of the data.")
      return expected, False
    return expected, (None if lower else True)

# --------------------------------------------------------------------------- layout classifier
class LayoutClassifier:
  """Entropy (information-gain) decision tree over column-name tokens + shape tokens."""

  # Used only when no training CSV is available. Shape descriptors alone carry the
  # signal (column names add nothing significant in the ablation), so the fallback
  # contains no domain vocabulary.
  FALLBACK = [
      ("__temporal __numeric_1 __one_row_per_key", "Line"),
      ("__highcard __numeric_1 __one_row_per_key", "Bar"),
      ("__geo __numeric_1 __one_row_per_key", "Geomap"),
      ("__numeric_2 __bounded __skewed", "Scatter"),
      ("__lowcard __numeric_1 __many_rows_per_key __bounded", "Boxplot"),
      ("__lowcard __numeric_1 __many_rows_per_key __skewed", "Violin"),
      ("__temporal __numeric_3plus", "Heatmap"),
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
      X = pd.Series([d for d, _ in self.FALLBACK])
      y = pd.Series([c for _, c in self.FALLBACK])
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
              random_state=self.clf.random_state,  # honours --seed
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

  def rules(self):
    """The fitted tree as IF-THEN rules over tokens, most-supported first."""
    from sklearn.tree import _tree
    t, names = self.clf.tree_, self.vec.get_feature_names_out()
    out = []

    def walk(node, conds):
      if t.feature[node] == _tree.TREE_UNDEFINED:
        counts = t.value[node][0]
        k = int(np.argmax(counts))
        n = int(t.n_node_samples[node])
        conf = float(counts[k] / counts.sum()) if counts.sum() else 0.0
        out.append(dict(layout=self.classes_[k], support=n, confidence=round(conf, 3),
                        conditions=" AND ".join(conds) or "TRUE"))
        return
      tok, thr = names[t.feature[node]], t.threshold[node]
      walk(t.children_left[node], conds + [f"{tok} absent" if thr < 1 else f"count({tok}) <= {thr:.1f}"])
      walk(t.children_right[node], conds + [f"{tok} present" if thr < 1 else f"count({tok}) > {thr:.1f}"])

    walk(0, [])
    return sorted(out, key=lambda r: -r["support"])

  def export_rules(self, path):
    rs = self.rules()
    with open(path, "w", encoding="utf-8") as f:
      f.write(f"# GraphLens layout rule base (decision tree: depth={self.clf.get_depth()}, "
              f"rules={len(rs)}, vocabulary={len(self.vec.vocabulary_)})\n")
      f.write("# Tokens starting with '__' are shape descriptors; others are column-name words.\n\n")
      for i, r in enumerate(rs, 1):
        f.write(f"R{i:02d}: IF {r['conditions']}\n     THEN {r['layout']}  "
                f"(support={r['support']}, confidence={r['confidence']:.2f})\n")
    return dict(depth=int(self.clf.get_depth()), n_rules=len(rs),
                vocabulary=len(self.vec.vocabulary_))

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
  full: pd.DataFrame = None  # complete table for renderers; routing uses df
  labels: dict = field(default_factory=dict)   # column -> axis/legend wording (from the profile)
  palette: dict = None                         # category value -> colour (same in every figure)
  order: list = None                           # category order (e.g. low -> high band)


# one colour per cluster band, assigned low -> high, identical in every figure
BAND_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


class GraphLensAgnosticPipeline:

  def __init__(
      self,
      profile_path_or_url,
      training_data_path="data/training_data.csv",
      output_dir="output",
      seed=42,
      stability_runs=None,
  ):
    self.output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)
    self.seed = seed
    self.timings, self.report = {}, {}
    self._load_profile(profile_path_or_url)
    if stability_runs is not None:  # CLI --stability-runs overrides profile
      self.cl["stability_runs"] = int(stability_runs)
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
            time="Year",
            count="Instance count",
            band="Cluster band",
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
        stability_runs=cfg.get("stability_runs", 50),  # default matches paper evaluation
    )
    self.filters = prof.get("filters", {})
    ana = prof.get("analysis", {})
    self.corr_method = ana.get("corr_method", "spearman")
    self.top_n = int(ana.get("top_n", 10))                    # bars in the top-entities view
    self.top_categories = int(ana.get("top_categories", 10))  # categories shown; the rest -> "Other"
    self.low_conf = float(ana.get("low_confidence", 0.5))     # routing confidence warning
    self.geo_style = str(ana.get("geomap", "map")).lower()  # "map" (choropleth) or "bar"
    self.geo_shapes = ana.get("geo_shapes", GEO_SHAPES_DEFAULT)
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
    self.entity_ids = {i.get("identifier") for i in attrs.get("entity_identifiers", [])}
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
    nrm = prof.get("normalization", {})
    scale = str(nrm.get("ratio_scale", "auto")).lower()
    if scale not in RATIO_SCALES:
      raise ValueError(f"normalization.ratio_scale must be one of {RATIO_SCALES}, got {scale!r}")
    self.norm = dict(
        ratio_scale=scale,
        year_re=re.compile(nrm.get("year_pattern", DEFAULT_YEAR_PATTERN)),
        time_fallback=list(nrm.get("time_fallback", ["date", "year"])),
        # common naming predicates (rdfs:label, foaf:name, dc:title, ...) used when
        # the profile declares no entity_identifiers
        entity_fallback=list(nrm.get("entity_fallback", ["label", "name", "title", "acronym"])),
    )
    self.numeric_ids = list(dict.fromkeys(
        ids["independent_metrics"] + ids["derived_ratios"]
        + [it["identifier"] for items in attrs.values() for it in items
           if it.get("identifier") and it.get("semantic_type") in ("numeric", "count", "ratio")]
    ))
    log.info("Profile loaded (target class: %s)", prof.get("target_class", "n/a"))

  @staticmethod
  def _suffix(val):
    s = str(val).strip()
    return s.split("/")[-1].split("#")[-1] if ("/" in s or "#" in s) else s

  @staticmethod
  def _series_name(val, year=None):
    """'ICSE 2019' -> 'ICSE' when the trailing 4-digit token is within one year of
    the row's own date (editions held early in the following year, e.g. FOCS 2021
    in Feb 2022). Unrelated trailing numbers such as 'ISO 9001' are kept."""
    if pd.isna(val):
      return np.nan
    s = str(val).strip()
    m = re.search(r"[\s_\-']*(\d{4})$", s)
    if m and year is not None and pd.notna(year) and abs(int(m.group(1)) - int(year)) <= 1:
      s = s[: m.start()].strip() or s
    return s

  def normalize(self, raw):
    rep = {"rows_raw": int(len(raw))}
    df = raw.rename(
        columns={k: v for k, v in self.var_map.items() if k in raw.columns}
    ).copy()

    for c in dict.fromkeys([self.k_vol, self.k_suc, self.k_rate] + self.numeric_ids):
      if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    tcol = next(
        (c for c in [self.k_time] + self.norm["time_fallback"] if c in df.columns), None
    )
    yre = self.norm["year_re"]

    def _year(x):
      if pd.isna(x):
        return np.nan
      m = yre.search(str(x))
      return int(m.group(1) if m.groups() else m.group()) if m else np.nan

    df["year"] = df[tcol].map(_year) if tcol else np.nan

    for c in df.columns:
      if c not in ("year", tcol) and not pd.api.types.is_numeric_dtype(df[c]):
        # namespace suffix; empty strings (e.g. an empty dbp:country literal) are missing
        df[c] = df[c].map(lambda x: (self._suffix(x) or np.nan) if pd.notna(x) else np.nan)
    for ident, amap in self.aliases.items():
      if ident in df.columns:
        df[ident] = df[ident].replace(amap)
    for ident in (self.k_cat, self.k_geo):
      if ident in df.columns:
        df[ident] = df[ident].map(split_identifier)
        # aliases may also be written in display form ('British Hong Kong' for British_Hong_Kong)
        amap = self.aliases.get(ident, {})
        if amap:
          df[ident] = df[ident].replace({split_identifier(k): v for k, v in amap.items()})
    ent = (
        self.k_ent
        if self.k_ent in df.columns
        else next((c for c in self.norm["entity_fallback"] if c in df.columns), None)
    )
    if ent:
      grp = pd.Series([self._series_name(v, yr) for v, yr in zip(df[ent], df["year"])],
                      index=df.index)
      # names differing only in letter case are one entity ('ICSE', 'Icse'); it is shown
      # in its most frequent spelling (ties: alphabetical), not forced to upper case
      key = grp.str.upper()
      spell = (pd.DataFrame({"k": key, "s": grp}).dropna().value_counts()
               .reset_index().sort_values(["k", "count", "s"], ascending=[True, False, True])
               .drop_duplicates("k").set_index("k")["s"])
      df["entity_group"] = key.map(spell)
      df["entity_label"] = [
          (
              f"{a} {int(y)}"
              if pd.notna(a) and pd.notna(y)
              else str(o)
          )
          for a, y, o in zip(df["entity_group"], df["year"], df[ent])
      ]

    num = [c for c in (self.k_vol, self.k_suc, self.k_rate) if c in df.columns]
    rep["missing_before"] = {c: int(df[c].isna().sum()) for c in num}

    x1, x2, y = self.k_vol, self.k_suc, self.k_rate
    have = x1 in df.columns and x2 in df.columns and y in df.columns
    for f in ("y_rescaled", "y_recovered", "x2_recovered", "inconsistent"):
      df[f] = False
    scale = self.norm["ratio_scale"]
    rep["ratio_scale"] = scale
    if y in df.columns:
      if scale in ("auto", "percent"):
        m = df[y] > 1.0 if scale == "auto" else df[y].notna()
        df.loc[m, y] = df.loc[m, y] / 100.0
        df.loc[m, "y_rescaled"] = True
      if scale != "unbounded":
        bad = (df[y] < 0) | (df[y] > 1)
        rep["y_out_of_range_set_missing"] = int(bad.sum())
        df.loc[bad, y] = np.nan
    if have and scale != "unbounded":  # x2 <= x1 only holds for part/whole ratios
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

    self.entropy_profile(df)
    n0 = len(df)
    _miss_year = df["year"].isna()
    _miss_x1 = df[x1].isna() if x1 in df.columns else pd.Series(False, index=df.index)
    _dropped = _miss_year | _miss_x1
    if _dropped.any():
      _audit = df[_dropped].copy()
      _audit["drop_reason"] = "missing_year"
      if x1 in df.columns:
        _audit.loc[_miss_x1 & ~_miss_year, "drop_reason"] = f"missing_{x1}"
        _audit.loc[_miss_year & _miss_x1, "drop_reason"] = f"missing_year_and_{x1}"
      _id_col = next(
          (c for c in ("entity_label", "entity_group", x1) if c in _audit.columns),
          None,
      )
      _acols = ([_id_col] if _id_col else []) + ["year", x1, "drop_reason"]
      _audit[[c for c in _acols if c in _audit.columns]].to_csv(
          os.path.join(self.output_dir, "normalization_dropped.csv"), index=False
      )
      log.info("Wrote %d dropped-row entries to normalization_dropped.csv", _dropped.sum())
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
    # canonical row order, so results do not depend on the order a file parser or an
    # endpoint (pages, partitions) happened to return the rows in
    key = pd.Series("", index=df.index)
    for c in df.columns:
      key = key + "\x1f" + df[c].astype(object).where(df[c].notna(), "").map(str)
    return df.iloc[np.argsort(key.values, kind="stable")].reset_index(drop=True)

  def entropy_profile(self, df, bins=10):
    """Per-attribute Shannon entropy H(X) (Eq. 1) as a schema/data-quality diagnostic.

    Categorical attributes use their value distribution; numeric attributes are
    discretised into equal-frequency bins. Low normalised entropy flags near-constant
    predicates; high entropy with mostly unique values flags identifier-like ones.
    """
    rows = []
    for ident in dict.fromkeys(list(self.display) + ["year"]):
      if ident not in df.columns:
        continue
      s = df[ident]
      nn = s.dropna()
      numeric = pd.api.types.is_numeric_dtype(s) and ident != "year"
      vals = (pd.qcut(nn, q=min(bins, max(nn.nunique(), 1)), duplicates="drop")
              if numeric and nn.nunique() > 1 else nn)
      h_bits = shannon_entropy(vals, normalize=False) if len(nn) else 0.0
      h_norm = shannon_entropy(vals, normalize=True) if len(nn) else 0.0
      miss = float(s.isna().mean()) if len(s) else 0.0
      uniq = nn.nunique() / len(nn) if len(nn) else 0.0
      flags = []
      if nn.nunique() <= 1 or h_norm < 0.1:
        flags.append("near-constant")
      if not numeric and uniq > 0.9 and len(nn) > 20:
        flags.append("identifier-like")
      if miss > 0.5:
        flags.append("sparse")
      rows.append(dict(attribute=ident, kind="numeric" if numeric else "categorical",
                       n_distinct=int(nn.nunique()), missing_rate=round(miss, 3),
                       entropy_bits=round(h_bits, 3), entropy_norm=round(h_norm, 3),
                       flags=";".join(flags)))
    prof = pd.DataFrame(rows)
    prof.to_csv(os.path.join(self.output_dir, "attribute_entropy.csv"), index=False)
    self.report["entropy_profile"] = prof.to_dict(orient="records")
    flagged = prof[prof["flags"] != ""] if len(prof) else prof
    for _, r in flagged.iterrows():
      if r["flags"] == "identifier-like" and r["attribute"] in self.entity_ids:
        continue                              # declared identifiers are meant to be unique
      log.warning("Attribute '%s' flagged %s (H_norm=%.2f, missing=%.0f%%)",
                  r["attribute"], r["flags"], r["entropy_norm"], 100 * r["missing_rate"])
    return prof

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
    if self.cl["fixed_k"] and int(self.cl["fixed_k"]) not in models:
      raise ValueError(
          f"fixed_k={self.cl['fixed_k']} was not evaluated "
          f"(swept range {self.cl['k_min']}–{kmax}). "
          "Set fixed_k to a value within that range, or remove it to use automatic selection."
      )
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
    # canonical ids: 0 = lowest mean-ratio band, so labels do not depend on row order
    df.loc[sub.index, "cluster_label"] = df.loc[sub.index, "cluster_label"].map(
        {c: i for i, c in enumerate(order)})
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
    nan = float("nan")
    if len(t) < 3 or np.ptp(t) == 0:        # no time variation: slope is undefined
      return dict(n_years=int(len(np.unique(t))), slope=nan, intercept=nan, ci95=[nan, nan],
                  p_value=nan, p_value_hac=nan, r2=nan, durbin_watson=nan, resid_var=nan,
                  note="undefined: fewer than 3 distinct years")
    X = sm.add_constant(t)
    with np.errstate(divide="ignore", invalid="ignore"):
      m = sm.OLS(v, X).fit()
      hac = sm.OLS(v, X).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
      flat = np.ptp(v) == 0                  # constant series: perfect but uninformative fit
      return dict(
          n_years=int(len(t)),
          slope=float(m.params[1]),
          intercept=float(m.params[0]),
          ci95=[float(x) for x in m.conf_int()[1]],
          p_value=nan if flat else float(m.pvalues[1]),
          p_value_hac=nan if flat else float(hac.pvalues[1]),
          r2=nan if flat else float(m.rsquared),
          durbin_watson=nan if flat else float(durbin_watson(m.resid)),
          resid_var=float(m.scale),
          **({"note": "constant series"} if flat else {}),
      )

  def trends(self, df):
    x1, out = self.k_vol, {}
    agg = (
        df.groupby("year")[x1]
        .agg(total="sum", per_entity="mean", n_entities="size")
        .reset_index()
    )
    agg.to_csv(
        os.path.join(self.output_dir, "yearly_aggregates.csv"), index=False
    )
    if len(agg) > 2:
      out["global_total"] = self._ols(agg["year"], agg["total"])
      out["global_per_entity_mean"] = self._ols(agg["year"], agg["per_entity"])
      out["global_n_entities"] = self._ols(agg["year"], agg["n_entities"])
      ll = self._ols(agg["year"], np.log(agg["total"].clip(lower=1)))
      out["global_total_loglinear"] = dict(
          ll, annual_growth_pct=float((np.exp(ll["slope"]) - 1) * 100)
      )
    per = {}
    for prof, g in df[df.cluster_label >= 0].groupby("agnostic_profile"):
      a = (
          g.groupby("year")[x1]
          .agg(total="sum", per_entity="mean")
          .reset_index()
      )
      if len(a) > 2:
        per[prof] = dict(
            total=self._ols(a["year"], a["total"]),
            per_entity_mean=self._ols(a["year"], a["per_entity"]),
        )
    out["per_cluster"] = per
    r2 = [v["total"]["r2"] for v in per.values()]
    r2 = [x for x in r2 if not np.isnan(x)]
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
    L = self.L
    # wording for every column a view can contain (column names stay as they are,
    # because they are also the layout classifier's input)
    lab = {d(x1): L["volume"], d(x2): L["success"], d(y): L["rate"], d(c1): L["category"],
           d(c2): L["location"], "year": L["time"], "clusterProfile": L["band"],
           "count": L["count"], "entity_group": L["entity_group"]}
    with_lab = lambda **over: {**lab, **over}
    V = []
    cl = df[df.cluster_label >= 0]
    bands = cl.sort_values("cluster_label")["agnostic_profile"].drop_duplicates().tolist()
    band_style = dict(palette={b: BAND_COLORS[i % len(BAND_COLORS)] for i, b in enumerate(bands)},
                      order=bands)
    if y in df.columns and len(cl):
      V.append(
          View(
              "cluster_scatter",
              cl[[x1, y, "agnostic_profile"]].rename(
                  columns={x1: d(x1), y: d(y), "agnostic_profile": "clusterProfile"}
              ),
              f"{L['volume']} vs {L['rate']} by cluster band",
              "fig1_cluster_scatter.png",
              "Scatter",
              labels=with_lab(),
              **band_style,
          )
      )
    if len(cl):
      tp, fits = {}, {}
      for prof, g in cl.groupby("agnostic_profile"):
        tp[prof] = g.groupby("year")[x1].sum()
        if prof in trend.get("per_cluster", {}):
          r = trend["per_cluster"][prof]["total"]
          if not np.isnan(r["slope"]):
            fits[prof] = (r["slope"], r["intercept"])
      tdf = pd.concat(tp, names=["clusterProfile", "year"]).reset_index(
          name=d(x1)
      )[["year", "clusterProfile", d(x1)]]
      V.append(
          View(
              "trend_by_cluster",
              tdf,
              f"Yearly {L['volume']} with per-cluster OLS fits",
              "fig2_cluster_trend_ols.png",
              "Line",
              time="year",
              fits=fits,
              labels=with_lab(**{d(x1): f"Yearly total {L['volume']}"}),
              **band_style,
          )
      )
    if c1 in df.columns:
      # charts stay readable: the most frequent categories are kept, the rest become "Other"
      counts = df[c1].value_counts()
      if len(counts) > self.top_categories:
        keep = set(counts.index[: self.top_categories])
        log.info("%s: %d categories; showing the %d most frequent, the rest as 'Other'",
                 L["category"], len(counts), self.top_categories)
        df = df.assign(**{c1: df[c1].where(df[c1].isin(keep) | df[c1].isna(), "Other")})
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
              f"Yearly {L['volume']} by {L['category']}",
              "fig3_field_growth.png",
              "Line",
              time="year",
              stacked=True,
              labels=with_lab(**{d(x1): f"Yearly total {L['volume']}"}),
          )
      )
    if c2 in df.columns:
      g_all = (
          df.groupby(c2)
          .size()
          .sort_values(ascending=False)
          .reset_index(name="count")
          .rename(columns={c2: d(c2)})
      )
      g = g_all.head(12)  # routing input unchanged; the map draws every location
      V.append(
          View(
              "geo_distribution",
              g,
              f"Distribution by {L['location']}",
              "fig4_geo_distribution.png",
              "Geomap",
              geo=d(c2),
              full=g_all,
              labels=with_lab(),
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
              labels=with_lab(),
          )
      )
    if c1 in df.columns and y in df.columns:
      sub = df[[c1, y]].dropna().rename(columns={c1: d(c1), y: d(y)})
      V.append(
          View(
              "ratio_by_category",
              sub,
              f"{L['rate']} by {L['category']}",
              "fig6_ratio_by_category.png",
              "Boxplot",
              labels=with_lab(),
          )
      )
    if "entity_group" in df.columns:
      t = (
          df.groupby("entity_group")[x1]
          .sum()
          .sort_values(ascending=False)
          .head(self.top_n)
          .reset_index()
          .rename(columns={x1: d(x1)})
      )
      # entity names are shown as written (NeurIPS, not 'Neur IPS'); IRI underscores become spaces
      t["entity_group"] = t["entity_group"].astype(str).str.replace("_", " ", regex=False)
      V.append(
          View(
              "top_entities",
              t,
              f"Top {self.top_n} {L['entity_group']} by cumulative {L['volume']}",
              "fig7_top_entities.png",
              "Bar",
              labels=with_lab(**{d(x1): f"Cumulative {L['volume']}"}),
          )
      )
    if len(cl):
      V.append(
          View(
              "cluster_density",
              cl[["agnostic_profile", x1]].rename(
                  columns={"agnostic_profile": "clusterProfile", x1: d(x1)}
              ),
              f"{L['volume']} distribution by cluster band",
              "fig8_cluster_density.png",
              "Violin",
              labels=with_lab(),
              **band_style,
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
      _conf = ranked[0][1]
      _LOW_CONF = self.low_conf
      if _conf < _LOW_CONF:
        log.warning(
            "View '%s': classifier confidence %.2f < %.2f for predicted layout '%s' — "
            "routing may be unreliable. Add an 'expected' layout in the profile to override.",
            v.name, _conf, _LOW_CONF, pred,
        )
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
              confidence=round(_conf, 3),
              low_confidence=(_conf < _LOW_CONF),
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

  @staticmethod
  def _lab(v, col):
    """Axis/legend wording for a column: the profile label, else a readable column name."""
    return v.labels.get(col) or split_identifier(col)

  @staticmethod
  def _cat_style(v, values):
    """(palette or None, order) for drawn category values. Keys are normalised the same
    way as the drawn values, so profile spelling (e.g. letter case) cannot break them."""
    present = [x for x in pd.unique(values) if pd.notna(x)]
    want = [split_identifier(k) for k in (v.order or [])]
    order = [o for o in want if o in present] + sorted(
        (x for x in present if x not in want), key=lambda x: (x == "Other", str(x)))
    pal = {split_identifier(k): c for k, c in (v.palette or {}).items()}
    return (pal if pal and all(o in pal for o in order) else None), order

  def _r_line(self, v, ax):
    num, cat = self._split(v)
    d = v.df.copy()
    if cat:
      d[cat[0]] = d[cat[0]].map(split_identifier)

    if cat and v.stacked:
      piv = d.pivot_table(
          index=v.time, columns=cat[0], values=num[0], aggfunc="sum"
      ).fillna(0)
      pal, order = self._cat_style(v, d[cat[0]])
      piv = piv[[c for c in order if c in piv.columns]]
      style = {"color": [pal[c] for c in piv.columns]} if pal else {"cmap": "tab20"}
      piv.plot(kind="area", stacked=True, ax=ax, alpha=0.85, **style)
      h, l = ax.get_legend_handles_labels()
      ax.legend(h, [_short(x) for x in l], title=self._lab(v, cat[0]), fontsize=8)
    elif cat:
      pal, order = self._cat_style(v, d[cat[0]])
      if len(order) > 12:
        log.warning("Line chart '%s': %d categories exceed 12; colors will repeat. "
                    "Consider filtering to top categories.", v.name, len(order))
      default = sns.color_palette("tab20" if len(order) > 9 else "Set1", min(max(len(order), 1), 20))
      fits = {split_identifier(k): f for k, f in v.fits.items()}
      for i, k in enumerate(order):
        g = d[d[cat[0]] == k].sort_values(v.time)
        col = pal[k] if pal else default[i % len(default)]
        ax.plot(g[v.time], g[num[0]], marker="o", ms=4, color=col, label=k)
        if k in fits:
          s, b = fits[k]
          xs = np.array([g[v.time].min(), g[v.time].max()])
          ax.plot(xs, b + s * xs, "--", color=col, lw=1)
      h, l = ax.get_legend_handles_labels()
      ax.legend(h, [_short(x) for x in l], title=self._lab(v, cat[0]), fontsize=8)
    else:
      ax.plot(d[v.time], d[num[0]], marker="o")

    if v.time == "year":
      ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    if max(abs(y) for y in ax.get_ylim()) >= 1e5:
      ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_compact))
    ax.set_xlabel(self._lab(v, v.time))
    ax.set_ylabel(self._lab(v, num[0]))

  @staticmethod
  def _shorten_ticks(ax, axis):
    """Shorten long category tick labels in place (display only)."""
    a = ax.yaxis if axis == "y" else ax.xaxis
    ticks, labels = a.get_ticklocs(), [t.get_text() for t in a.get_ticklabels()]
    if any(len(t) > 32 for t in labels):
      a.set_ticks(ticks)
      a.set_ticklabels([_short(t) for t in labels])

  def _r_bar(self, v, ax):
    num, cat = self._split(v)
    d = v.df.sort_values(num[0], ascending=False).head(12).copy()
    if v.name != "top_entities":
      d[cat[0]] = d[cat[0]].map(split_identifier)
    sns.barplot(data=d, x=num[0], y=cat[0], hue=cat[0], palette="flare", ax=ax)
    if ax.get_legend():
      ax.get_legend().remove()
    self._shorten_ticks(ax, "y")
    if max(abs(x) for x in ax.get_xlim()) >= 1e5:
      ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_compact))
    ax.set_xlabel(self._lab(v, num[0]))
    ax.set_ylabel(self._lab(v, cat[0]))

  def _world(self):
    if not hasattr(self, "_world_cache"):
      self._world_cache = load_world_shapes(self.geo_shapes)
    return self._world_cache

  def _r_geo(self, v, ax):
    """Choropleth world map; falls back to a ranked bar chart if the basemap is
    missing, the profile sets analysis.geomap: bar, or <50% of the values match."""
    world = self._world() if self.geo_style != "bar" else None
    if world is None:
      return self._r_geo_bar(v, ax)
    num, cat = self._split(v)
    loc, val = cat[0], num[0]
    d = (v.full if v.full is not None else v.df)[[loc, val]].dropna()
    ids = d[loc].map(lambda n: _geo_lookup(world, n))
    total = float(d[val].sum())
    coverage = float(d.loc[ids.notna(), val].sum()) / total if total else 0.0
    unmatched = sorted(d.loc[ids.isna(), loc].astype(str).unique().tolist())
    self.report["geomap"] = dict(
        locations=int(len(d)),
        matched=int(ids.notna().sum()),
        value_coverage=round(coverage, 3),
        unmatched=unmatched,
    )
    if coverage < 0.5:
      log.warning(
          "Geomap: only %.0f%% of values matched a country -> bar chart instead."
          " Unmatched: %s", 100 * coverage, ", ".join(unmatched[:10]),
      )
      return self._r_geo_bar(v, ax)
    if unmatched:
      log.warning("Geomap: %d location(s) not on the basemap: %s",
                  len(unmatched), ", ".join(unmatched[:10]))
    m = d.assign(_id=ids).dropna(subset=["_id"])
    vals = m.groupby("_id")[val].sum()
    vals = vals[vals > 0]
    label_of = m.drop_duplicates("_id").set_index("_id")[loc].map(split_identifier)

    lo, hi = float(vals.min()), float(vals.max())
    norm = (mcolors.LogNorm(vmin=lo, vmax=hi) if hi / lo >= 20
            else mcolors.Normalize(vmin=0, vmax=hi))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "graphlens_geo", plt.cm.Blues(np.linspace(0.25, 1.0, 256)))
    nodata, has, z = [], [], []
    for cid, path in world["paths"].items():
      if cid in vals.index:
        has.append(PathPatch(path)); z.append(vals[cid])
      else:
        nodata.append(PathPatch(path))
    ax.add_collection(PatchCollection(nodata, facecolor="#e6e6e6", edgecolor="white", linewidth=0.3))
    pc = PatchCollection(has, cmap=cmap, norm=norm, edgecolor="white", linewidth=0.3)
    pc.set_array(np.asarray(z, float))
    ax.add_collection(pc)

    # countries too small to see as polygons (e.g. Singapore, Luxembourg) get a dot
    small = [c for c in vals.index if world["area"].get(c, 0) < 1.5]
    if small:
      xy = np.array([world["anchor"][c] for c in small])
      ax.scatter(xy[:, 0], xy[:, 1], c=vals[small].values, cmap=cmap, norm=norm,
                 s=22, edgecolors="white", linewidths=0.8, zorder=3)

    # direct labels for the largest values, skipping any that would collide with one
    # already placed (dense regions such as Europe); a corner box lists the top five
    halo = [pe.withStroke(linewidth=2.5, foreground="white")]
    top = vals.sort_values(ascending=False)
    placed = []
    for cid in top.index[:6]:
      x, y = world["anchor"][cid]
      if len(placed) == 3 or any(abs(x - px) < 22 and abs(y - py) < 12 for px, py in placed):
        continue
      placed.append((x, y))
      ax.annotate(f"{label_of.get(cid, world['names'][cid])}\n{vals[cid]:,.0f}", (x, y), ha="center",
                  va="center", fontsize=7.5, color="#222222", path_effects=halo, zorder=4)
    lines = [f"{i}. {label_of.get(c, world['names'][c])}  {top[c]:,.0f}"
             for i, c in enumerate(top.index[:5], 1)]
    ax.text(0.01, 0.03, "Top 5\n" + "\n".join(lines), transform=ax.transAxes, fontsize=7,
            va="bottom", ha="left", color="#222222", zorder=5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.9))

    ax.figure.set_size_inches(10, 5.2)  # world-map aspect; avoids empty bands
    ax.set_xlim(-170, 180)
    ax.set_ylim(-57, 84)
    ax.set_aspect("equal")
    ax.set_axis_off()
    cb = ax.figure.colorbar(pc, ax=ax, orientation="horizontal", fraction=0.05,
                            pad=0.02, shrink=0.55, aspect=35)
    cb.outline.set_visible(False)
    if isinstance(norm, mcolors.LogNorm):   # readable ticks: 1, 2, 5, 10, 20, 50, ...
      cb.ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
      cb.ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    cb.ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}" if x >= 1 else f"{x:g}"))
    cb.ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    cb.set_label(self._lab(v, val)
                 + (" (log scale)" if isinstance(norm, mcolors.LogNorm) else ""))

  def _r_geo_bar(self, v, ax):
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
    self._shorten_ticks(ax, "y")
    ax.set_xlabel(self._lab(v, num[0]))
    ax.set_ylabel(self._lab(v, cat[0]))

  def _r_scatter(self, v, ax):
    num, cat = self._split(v)
    d = v.df.copy()
    pal, order = None, None
    if cat:
      d[cat[0]] = d[cat[0]].map(split_identifier)
      pal, order = self._cat_style(v, d[cat[0]])
    sns.scatterplot(
        data=d,
        x=num[0],
        y=num[1],
        hue=cat[0] if cat else None,
        hue_order=order,
        palette=(pal or "Set1") if cat else None,
        s=60,
        alpha=0.8,
        edgecolor="black",
        ax=ax,
    )
    if cat and ax.get_legend():
      leg = ax.get_legend()
      leg.set_title(self._lab(v, cat[0]))
      for t in leg.get_texts():
        t.set_text(_short(t.get_text()))
    if d[num[0]].max() > 500:
      ax.set_xscale("log")
    ax.set_xlabel(self._lab(v, num[0]) + (" (log scale)" if ax.get_xscale() == "log" else ""))
    ax.set_ylabel(self._lab(v, num[1]))

  def _r_box(self, v, ax):
    num, cat = self._split(v)
    if not num or not cat:
      return
    d = v.df.copy()
    d[cat[0]] = d[cat[0]].map(split_identifier)
    s = d[num[0]].dropna()
    is_ratio = (s.min() >= 0) and (s.max() <= 1.0)
    pal, order = self._cat_style(v, d[cat[0]])

    sns.boxplot(data=d, x=num[0], y=cat[0], hue=cat[0], order=order, hue_order=order,
                palette=pal or "vlag", ax=ax)
    if ax.get_legend():
      ax.get_legend().remove()
    self._shorten_ticks(ax, "y")

    if is_ratio:
      ax.xaxis.set_major_formatter(
          matplotlib.ticker.PercentFormatter(xmax=1.0)
      )
    ax.set_xlabel(self._lab(v, num[0]))
    ax.set_ylabel(self._lab(v, cat[0]))

  def _r_violin(self, v, ax):
    num, cat = self._split(v)
    if not num or not cat:
      return
    d = v.df.copy()
    d[cat[0]] = d[cat[0]].map(split_identifier)
    s = d[num[0]].dropna()
    pal, order = self._cat_style(v, d[cat[0]])

    is_ratio = (s.min() >= 0) and (s.max() <= 1.0)
    is_skewed = (s.min() > 0) and (
        s.max() / max(s.min(), 1e-5) > 100 or abs(sps.skew(s)) > 1.5
    )
    use_log = bool(is_skewed and not is_ratio)

    # on a log axis the ticks show real values (1, 10, 100, ...), not log10 numbers
    sns.violinplot(
        data=d, x=cat[0], y=num[0], hue=cat[0], order=order, hue_order=order,
        palette=pal or "Pastel1", cut=0, log_scale=use_log, ax=ax
    )
    if ax.get_legend():
      ax.get_legend().remove()

    if is_ratio and not use_log:
      ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    elif use_log:
      ax.yaxis.set_major_formatter(
          matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}" if x >= 1 else f"{x:g}"))

    self._shorten_ticks(ax, "x")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    ax.set_xlabel(self._lab(v, cat[0]))
    ax.set_ylabel(self._lab(v, num[0]) + (" (log scale)" if use_log else ""))

  def _r_heat(self, v, ax):
    corr = v.df.corr(method=self.corr_method, numeric_only=True)
    names = [self._lab(v, c) for c in corr.columns]
    corr.index, corr.columns = names, names
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.7,
        ax=ax,
    )
    # long names go onto two lines so neighbouring column labels do not overlap
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels([textwrap.fill(n, 14) for n in names], rotation=0)

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
    self.report.setdefault("routing", {})["rule_base"] = self.layout.export_rules(
        os.path.join(self.output_dir, "layout_rules.txt"))
    df.to_csv(os.path.join(self.output_dir, "features_matrix.csv"), index=False)
    self.report["timings_s"] = {
        k: round(v, 3) for k, v in self.timings.items()
    }
    with open(
        os.path.join(self.output_dir, "run_report.json"), "w", encoding="utf-8"
    ) as f:
      json.dump(self.report, f, indent=2, default=str)
    return df

  def execute_pipeline(self, query_str, rdf_url=None, rdf_file=None, endpoint=None, **endpoint_opts):
    return self.run_from_dataframe(
        self.query_to_dataframe(query_str, rdf_url, rdf_file, endpoint, **endpoint_opts))

  def query_to_dataframe(self, query_str, rdf_url=None, rdf_file=None, endpoint=None, **endpoint_opts):
    """Return the raw SPARQL projection as a DataFrame, from a SPARQL endpoint
    (paged; nothing is loaded into memory except the result) or a parsed RDF file."""
    if endpoint:
      t0 = time.perf_counter()
      with self._timer("module1_ingest_and_sparql"):
        client = SparqlEndpoint(endpoint, **endpoint_opts)
        raw = client.select(query_str)
      self.report["ingest"] = dict(client.stats, seconds=round(time.perf_counter() - t0, 3))
      st = client.stats
      sliced = (f"{st['sample_partitions']} of {st['partitions']} slices, "
                if st.get("sample_partitions") else
                f"{st['partitions']} slices, " if st["partitions"] > 1 else "")
      log.info("Retrieved %s rows in total from %s (%s%d page(s), %d request(s), %.0f s)",
               f"{st['rows']:,}", endpoint, sliced, st["pages"], st["requests"],
               time.perf_counter() - t0)
      if raw.empty:
        raise ValueError("SPARQL projection returned no rows.")
      return raw
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
    return raw


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
  ap.add_argument(
      "--stability-runs",
      type=int,
      default=None,
      metavar="N",
      help="bootstrap stability runs for clustering (overrides profile; default 50)",
  )
  ap.add_argument("--log-level", default="INFO")
  g = ap.add_mutually_exclusive_group(required=True)
  g.add_argument("-u", "--url", help="RDF file to download and parse in memory")
  g.add_argument("-f", "--file", help="local RDF file to parse in memory")
  g.add_argument("-e", "--endpoint", help="SPARQL endpoint URL; the query runs remotely, paged")
  ep = ap.add_argument_group("endpoint options (with -e)")
  ep.add_argument("--page-size", type=int, default=5000, help="rows per request (default 5000)")
  ep.add_argument("--max-rows", type=int, default=None, help="stop after this many rows; with a GRAPHLENS-PARTITION query, "
                  "whole slices are fetched, giving a reproducible random sample")
  ep.add_argument("--timeout", type=int, default=120, help="seconds per request (default 120)")
  ep.add_argument("--stable-order", action="store_true",
                  help="add ORDER BY over all variables so pages are deterministic (slower)")
  ep.add_argument("--partitions", type=int, default=16, choices=(1, 16, 256),
                  help="slices for queries marked '# GRAPHLENS-PARTITION ?var' (default 16)")
  ep.add_argument("--no-count-check", action="store_true",
                  help="skip the COUNT query that verifies the endpoint returned every row")
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
    GraphLensAgnosticPipeline(
        a.profile, a.training, a.output, a.seed,
        stability_runs=a.stability_runs,
    ).execute_pipeline(
        q, rdf_url=a.url, rdf_file=a.file, endpoint=a.endpoint,
        **(dict(page_size=a.page_size, max_rows=a.max_rows, timeout=a.timeout,
                stable_order=a.stable_order, count_check=not a.no_count_check,
                partitions=a.partitions)
           if a.endpoint else {}),
    )
  except Exception as e:
    log.error("%s: %s", type(e).__name__, e)
    sys.exit(1)


if __name__ == "__main__":
  main()