"""Unit and end-to-end tests for GraphLens (run with: python3 -m pytest)."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import graphlens as gl  # noqa: E402

PROFILE = """
target_class: Event
attributes:
  independent_metrics:
    - {identifier: submitted, sparql_var: submitted}
    - {identifier: accepted, sparql_var: accepted}
  derived_ratios:
    - {identifier: rate, sparql_var: rate}
  nominal_categories:
    - {identifier: field, sparql_var: field}
    - {identifier: country, sparql_var: country, semantic_type: geo}
  temporal_indicators:
    - {identifier: date, sparql_var: date}
  entity_identifiers:
    - {identifier: name, sparql_var: name}
"""


def make_pipeline(tmp_path, extra="", **kw):
  p = tmp_path / "profile.yaml"
  p.write_text(PROFILE + extra)
  return gl.GraphLensAgnosticPipeline(str(p), training_data_path=None,
                                      output_dir=str(tmp_path / "out"), **kw)


def synthetic(n=80, seed=1):
  rng = np.random.default_rng(seed)
  fields = ["ArtificialIntelligence", "Databases", "SoftwareEngineering", "Robotics"]
  countries = ["http://dbpedia.org/resource/United_States", "http://dbpedia.org/resource/Germany",
               "http://dbpedia.org/resource/Egypt", "http://dbpedia.org/resource/Japan"]
  rows = []
  for i in range(n):
    year = 2000 + i % 15
    sub = int(rng.integers(50, 3000))
    rate = round(float(rng.uniform(0.1, 0.6)), 3)
    rows.append(dict(name=f"CONF{i % 10} {year}", date=f"{year}-06-01", submitted=sub,
                     accepted=int(sub * rate), rate=rate, field=f"http://ex.org/{fields[i % 4]}",
                     country=countries[i % 4]))
  return pd.DataFrame(rows)


# ------------------------------------------------------------------ Eq. 1
def test_shannon_entropy_uniform_and_constant():
  s = pd.Series(list("abcd") * 5)
  assert gl.shannon_entropy(s, normalize=False) == pytest.approx(2.0)
  assert gl.shannon_entropy(s, normalize=True) == pytest.approx(1.0)
  assert gl.shannon_entropy(pd.Series(["x"] * 10)) == 0.0


# ------------------------------------------------------------------ normalisation
def test_recovers_ratio_and_count(tmp_path):
  pipe = make_pipeline(tmp_path)
  raw = pd.DataFrame(dict(date=["2010-01-01", "2011-01-01"], submitted=[200, 400],
                          accepted=[50, None], rate=[None, 0.25]))
  out = pipe.normalize(raw)
  assert out.loc[0, "rate"] == pytest.approx(0.25) and out.loc[0, "y_recovered"]
  assert out.loc[1, "accepted"] == 100 and out.loc[1, "x2_recovered"]


def test_inconsistent_count_is_flagged(tmp_path):
  pipe = make_pipeline(tmp_path)
  out = pipe.normalize(pd.DataFrame(dict(date=["2010"], submitted=[10], accepted=[20], rate=[None])))
  assert out.loc[0, "inconsistent"] and np.isnan(out.loc[0, "accepted"])


def test_generic_year_parsing(tmp_path):
  pipe = make_pipeline(tmp_path)
  raw = pd.DataFrame(dict(date=["1850-03-01", "2019-06-01T00:00:00", "held in 1999"],
                          submitted=[10, 20, 30], accepted=[1, 2, 3], rate=[None] * 3))
  assert pipe.normalize(raw)["year"].tolist() == [1850, 2019, 1999]


@pytest.mark.parametrize("scale,value,expected", [
    ("auto", 25.0, 0.25), ("percent", 0.5, 0.005), ("fraction", 1.5, None), ("unbounded", 1.5, 1.5)])
def test_ratio_scale_modes(tmp_path, scale, value, expected):
  pipe = make_pipeline(tmp_path, f"normalization: {{ratio_scale: {scale}}}\n")
  out = pipe.normalize(pd.DataFrame(dict(date=["2010"], submitted=[100], accepted=[None], rate=[value])))
  got = out.loc[0, "rate"]
  assert (np.isnan(got) if expected is None else got == pytest.approx(expected))


def test_invalid_ratio_scale_rejected(tmp_path):
  with pytest.raises(ValueError):
    make_pipeline(tmp_path, "normalization: {ratio_scale: weird}\n")


def test_dropped_rows_audit(tmp_path):
  pipe = make_pipeline(tmp_path)
  pipe.normalize(pd.DataFrame(dict(date=[None, "2010"], submitted=[5, None], accepted=[1, 1], rate=[None] * 2)))
  audit = pd.read_csv(tmp_path / "out" / "normalization_dropped.csv")
  assert set(audit["drop_reason"]) == {"missing_year", "missing_submitted"}


@pytest.mark.parametrize("label,year,expected", [
    ("ICSE 2019", 2019, "ICSE"), ("FOCS 2021", 2022, "FOCS"), ("ISO 9001", 2015, "ISO 9001")])
def test_series_name(label, year, expected):
  assert gl.GraphLensAgnosticPipeline._series_name(label, year) == expected


# ------------------------------------------------------------------ geo
@pytest.mark.skipif(not os.path.exists(gl.GEO_SHAPES_DEFAULT), reason="basemap not installed")
@pytest.mark.parametrize("name,code", [
    ("United_States", "USA"), ("UK", "GBR"), ("Georgia_(country)", "GEO"), ("Czech Republic", "CZE"),
    ("Türkiye", "TUR"), ("Singapore", "SGP"), ("Cinema_of_Ireland", "IRL"),
    ("Cinema Of The United States", "USA"), ("Weimar_Republic", "DEU"), ("Kingdom_of_France", "FRA"),
    ("Polish_People's_Republic", "POL"), ("Republic Of China (1912–1949)", None),
    ("Congo (Kinshasa)", "COD"), ("Congo (Brazzaville)", "COG"), ("Chinese Taipei", "TWN")])
def test_country_matching(name, code):
  world = gl.load_world_shapes(gl.GEO_SHAPES_DEFAULT)
  assert gl._geo_lookup(world, name) == code


# ------------------------------------------------------------------ classifier
def test_rule_base_export(tmp_path):
  clf = gl.LayoutClassifier(None, seed=0)
  rules = clf.rules()
  assert rules and all(r["layout"] in gl.LAYOUTS for r in rules)
  stats = clf.export_rules(str(tmp_path / "rules.txt"))
  assert stats["n_rules"] == len(rules) and (tmp_path / "rules.txt").read_text().count("THEN") == len(rules)


# ------------------------------------------------------------------ clustering
def test_fixed_k_outside_sweep_raises(tmp_path):
  pipe = make_pipeline(tmp_path, "clustering: {fixed_k: 15, stability_runs: 2}\n")
  df = pipe.normalize(synthetic())
  with pytest.raises(ValueError, match="fixed_k=15"):
    pipe.cluster(df)


def test_clustering_is_seed_reproducible(tmp_path):
  labels = []
  for run in range(2):
    (tmp_path / str(run)).mkdir()
    pipe = make_pipeline(tmp_path / str(run), seed=7, stability_runs=3)
    labels.append(pipe.cluster(pipe.normalize(synthetic()))["cluster_label"].tolist())
  assert labels[0] == labels[1]


# ------------------------------------------------------------------ end to end
def test_end_to_end(tmp_path):
  pipe = make_pipeline(tmp_path, stability_runs=3)
  pipe.run_from_dataframe(synthetic())
  out = tmp_path / "out"
  for f in ("run_report.json", "routing_report.csv", "features_matrix.csv", "cluster_validation.csv",
            "attribute_entropy.csv", "layout_rules.txt", "yearly_aggregates.csv"):
    assert (out / f).exists(), f
  rep = json.loads((out / "run_report.json").read_text())
  assert {"normalization", "clustering", "trends", "routing", "entropy_profile"} <= rep.keys()
  assert len(list(out.glob("fig*.png"))) >= 6


def test_ols_undefined_cases_are_nan_without_warnings():
  import warnings
  with warnings.catch_warnings():
    warnings.simplefilter("error")
    single = gl.GraphLensAgnosticPipeline._ols([2010, 2010, 2010], [1, 2, 3])
    flat = gl.GraphLensAgnosticPipeline._ols([2010, 2011, 2012, 2013], [5, 5, 5, 5])
    ok = gl.GraphLensAgnosticPipeline._ols([2010, 2011, 2012, 2013], [1, 2, 4, 8])
  assert np.isnan(single["slope"]) and np.isnan(flat["r2"]) and flat["slope"] == pytest.approx(0)
  assert ok["slope"] > 0 and 0 < ok["r2"] <= 1


def test_empty_strings_are_missing(tmp_path):
  pipe = make_pipeline(tmp_path)
  raw = pd.DataFrame(dict(date=["2010", "2011"], submitted=[10, 20], accepted=[1, 2], rate=[None] * 2,
                          country=["", "http://dbpedia.org/resource/"], field=["  ", "http://ex.org/AI"]))
  out = pipe.normalize(raw)
  assert out["country"].isna().all() and out["field"].isna().tolist() == [True, False]


def test_band_colours_and_labels_follow_the_profile(tmp_path):
  """Each band keeps one colour in every view, labels come from the profile, and band
  names in any letter case still get their trend fits."""
  extra = ("labels: {volume: Submissions, rate: Acceptance, band: Selectivity band}\n"
           "cluster_bands: [low band, mid band, high band]\nclustering: {fixed_k: 3, stability_runs: 2}\n")
  pipe = make_pipeline(tmp_path, extra)
  df = pipe.cluster(pipe.normalize(synthetic(150)))
  views = {v.name: v for v in pipe.build_views(df, pipe.trends(df)[1])}
  pal = views["cluster_scatter"].palette
  assert pal == views["trend_by_cluster"].palette == views["cluster_density"].palette
  assert views["cluster_scatter"].order == ["low band", "mid band", "high band"]
  v = views["cluster_scatter"]
  assert pipe._lab(v, "submitted") == "Submissions" and pipe._lab(v, "clusterProfile") == "Selectivity band"
  t = views["trend_by_cluster"]
  drawn = set(pd.unique(t.df["clusterProfile"].map(gl.split_identifier)))
  assert {gl.split_identifier(k) for k in t.fits} <= drawn and t.fits


def test_entropy_ignores_empty_categories():
  import warnings
  s = pd.Series(pd.Categorical(["a", "a", "b", "b"], categories=["a", "b", "c"]))
  with warnings.catch_warnings():
    warnings.simplefilter("error")
    assert gl.shannon_entropy(s, normalize=False) == pytest.approx(1.0)
    assert gl.shannon_entropy(s, normalize=True) == pytest.approx(1.0)


def test_rare_categories_fold_into_other_in_views_only(tmp_path):
  """With many categories the charts show the most frequent ones plus 'Other' (drawn last);
  the analysed data itself is not changed."""
  pipe = make_pipeline(tmp_path, "analysis: {top_categories: 2}\nclustering: {fixed_k: 3, stability_runs: 2}\n")
  df = pipe.cluster(pipe.normalize(synthetic(150)))
  before = df["field"].copy()
  views = pipe.build_views(df, pipe.trends(df)[1])
  assert df["field"].equals(before)
  shown = [v for v in views if "field" in v.df.columns]
  assert shown
  for v in shown:
    vals = set(v.df["field"].dropna().map(gl.split_identifier))
    assert len(vals) <= 3 and "Other" in vals
    assert pipe._cat_style(v, v.df["field"].map(gl.split_identifier))[1][-1] == "Other"


def test_display_helpers():
  assert gl._short("x" * 40).endswith("…") and len(gl._short("x" * 40)) == 32
  assert gl._short("Hindi") == "Hindi"
  assert [gl._compact(x) for x in (0, 500, 2e5, 7.5e8, 1.25e9)] == ["0", "500", "200k", "750M", "1.25B"]


def test_entity_names_keep_their_spelling(tmp_path):
  """Case variants are one entity, shown in the most frequent spelling (not upper case)."""
  pipe = make_pipeline(tmp_path)
  raw = synthetic(6).assign(name=["NeurIPS 2000", "NeurIPS 2001", "neurips 2002",
                                  "EuroSys 2003", "EuroSys 2004", "EuroSys 2005"],
                            date=[f"{y}-06-01" for y in range(2000, 2006)])
  out = pipe.normalize(raw)
  assert sorted(set(out["entity_group"])) == ["EuroSys", "NeurIPS"]


def test_country_alias_in_display_form_matches_iri(tmp_path):
  pipe = make_pipeline(tmp_path, 'value_aliases: {country: {"British Hong Kong": "Hong Kong"}}\n')
  raw = synthetic(4).assign(country="http://dbpedia.org/resource/British_Hong_Kong")
  assert set(pipe.normalize(raw)["country"]) == {"Hong Kong"}
