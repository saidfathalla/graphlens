"""Tests for SPARQL endpoint mode (offline: uses a local rdflib-backed server)."""
import os
import sys

import pandas as pd
import pytest
import rdflib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import graphlens as gl  # noqa: E402
from sparql_server import serve  # noqa: E402
from test_graphlens import PROFILE  # noqa: E402

TTL_HEAD = "@prefix ex: <http://ex.org/> .\n@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n"
QUERY = """PREFIX ex: <http://ex.org/>
# a comment with a quote ' and braces { }
SELECT ?e ?name ?date ?submitted ?accepted ?rate WHERE {
  ?e ex:name ?name ; ex:date ?date ; ex:submitted ?submitted .
  OPTIONAL { ?e ex:accepted ?accepted }
  OPTIONAL { ?e ex:rate ?rate }   # rate is optional
}"""


@pytest.fixture(scope="module")
def graph():
  lines = [TTL_HEAD]
  for i in range(137):
    y = 2000 + i % 20
    acc = "" if i % 5 == 0 else f' ; ex:accepted {10 + i}'
    rate = "" if i % 3 == 0 else f' ; ex:rate "0.{(i % 9) + 1}"^^xsd:decimal'
    lines.append(f'ex:e{i} ex:name "EV{i % 7} {y}" ; ex:date "{y}-05-0{1 + i % 9}"^^xsd:date ; '
                 f'ex:submitted {100 + 3 * i}{acc}{rate} .')
  g = rdflib.Graph()
  g.parse(data="\n".join(lines), format="turtle")
  return g


def file_frame(g, query=QUERY):
  res = g.query(query)
  return pd.DataFrame([[x.toPython() if x is not None else None for x in r] for r in res],
                      columns=[str(v) for v in res.vars])


def canon(df):
  return df.sort_values("e").reset_index(drop=True)


@pytest.mark.parametrize("page_size,row_cap", [(25, None), (50, 20), (1000, 30)])
def test_paging_matches_file_mode(graph, page_size, row_cap):
  url, stop, state = serve(graph, row_cap=row_cap)
  try:
    ep = gl.SparqlEndpoint(url, page_size=page_size)
    df = ep.select(QUERY)
  finally:
    stop()
  ref = file_frame(graph)
  assert len(df) == len(ref) == 137
  pd.testing.assert_frame_equal(canon(df), canon(ref))
  assert ep.stats["pages"] >= 137 // min(page_size, row_cap or page_size)
  pages = [q for q in state["queries"] if "COUNT(*)" not in q]
  assert pages and all("LIMIT" in q for q in pages)
  assert ep.stats["complete"] is True and ep.stats["expected_rows"] == 137


def test_stable_order_keeps_projection_order(graph):
  url, stop, _ = serve(graph)
  try:
    df = gl.SparqlEndpoint(url, page_size=40, stable_order=True).select(QUERY)
  finally:
    stop()
  assert list(df.columns) == ["e", "name", "date", "submitted", "accepted", "rate"]
  pd.testing.assert_frame_equal(canon(df), canon(file_frame(graph)))


def test_max_rows_and_existing_limit(graph):
  url, stop, state = serve(graph)
  try:
    assert len(gl.SparqlEndpoint(url, page_size=30, max_rows=70).select(QUERY)) == 70
    ep = gl.SparqlEndpoint(url, page_size=10)
    assert len(ep.select(QUERY + "\nLIMIT 15")) == 15 and ep.stats["paged"] is False
  finally:
    stop()


def test_retries_on_503(graph):
  url, stop, _ = serve(graph, fail_first=2)
  try:
    ep = gl.SparqlEndpoint(url, page_size=500, retries=3, backoff=0)
    # 2 failures + 1 page + 1 empty page + 1 COUNT verification
    assert len(ep.select(QUERY)) == 137 and ep.stats["requests"] == 5
  finally:
    stop()


def test_rejects_non_select(graph):
  with pytest.raises(ValueError):
    gl.SparqlEndpoint("http://127.0.0.1:9/none").select("CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }")


def test_comment_stripping_keeps_iris_and_strings():
  q = 'PREFIX a: <http://x/#a> # c\nSELECT ?s WHERE { ?s a:p "x # y" . FILTER(?o < 5) } # end'
  out = gl._strip_comments(q)
  assert "<http://x/#a>" in out and '"x # y"' in out and "?o < 5" in out and "# c" not in out


def test_pipeline_endpoint_equals_file(graph, tmp_path):
  url, stop, _ = serve(graph, row_cap=25)
  ttl = tmp_path / "g.ttl"
  graph.serialize(ttl, format="turtle")
  profile = tmp_path / "p.yaml"
  profile.write_text(PROFILE.replace("sparql_var: name}", "sparql_var: name}\n    - {identifier: e, sparql_var: e}")
                     + "clustering: {stability_runs: 2}\n")
  try:
    outs = {}
    for mode, kw in (("file", dict(rdf_file=str(ttl))), ("endpoint", dict(endpoint=url, page_size=40))):
      p = gl.GraphLensAgnosticPipeline(str(profile), None, str(tmp_path / mode), seed=3)
      outs[mode] = p.execute_pipeline(QUERY, **kw)
    assert p.report["ingest"]["rows"] == 137
  finally:
    stop()
  pd.testing.assert_frame_equal(outs["file"].sort_values("e").reset_index(drop=True),
                                outs["endpoint"].sort_values("e").reset_index(drop=True))


def test_incomplete_paging_is_detected(graph):
  url, stop, _ = serve(graph, window_cap=60)      # endpoint silently stops after 60 rows
  try:
    ep = gl.SparqlEndpoint(url, page_size=25)
    df = ep.select(QUERY)
  finally:
    stop()
  assert len(df) == 60
  assert ep.stats["expected_rows"] == 137 and ep.stats["complete"] is False


def test_partial_empty_page_is_retried_not_taken_as_end(graph):
  # DBpedia under load: a page comes back empty but flagged as partial (time limit hit)
  url, stop, _ = serve(graph, row_cap=30, partial_first=2)
  try:
    ep = gl.SparqlEndpoint(url, page_size=50, backoff=0)
    df = ep.select(QUERY)
  finally:
    stop()
  assert len(df) == 137 and ep.stats["complete"] is True
  assert ep.stats["incomplete_responses"] == 2


def test_timed_out_count_is_a_lower_bound(graph):
  url, stop, _ = serve(graph, partial_count=True)
  try:
    ep = gl.SparqlEndpoint(url, page_size=500, backoff=0)
    df = ep.select(QUERY)
  finally:
    stop()
  assert len(df) == 137 and ep.stats["expected_is_lower_bound"] is True
  assert ep.stats["complete"] is None          # cannot be verified


def test_count_check_can_be_disabled(graph):
  url, stop, state = serve(graph)
  try:
    ep = gl.SparqlEndpoint(url, page_size=500, count_check=False)
    ep.select(QUERY)
  finally:
    stop()
  assert ep.stats["expected_rows"] is None and not any("COUNT" in q for q in state["queries"])


def test_broken_transfer_is_retried(graph):
  url, stop, _ = serve(graph, break_first=1)       # first page is cut off mid-download
  try:
    ep = gl.SparqlEndpoint(url, page_size=500, backoff=0)
    df = ep.select(QUERY)
  finally:
    stop()
  assert len(df) == 137 and ep.stats["complete"] is True and ep.stats["failed_requests"] == 1


def test_page_size_shrinks_when_large_pages_keep_failing(graph):
  url, stop, _ = serve(graph, break_limit_above=40)  # any page over 40 rows is cut off
  try:
    ep = gl.SparqlEndpoint(url, page_size=200, min_page_size=10, backoff=0)
    df = ep.select(QUERY)
  finally:
    stop()
  assert len(df) == 137 and ep.stats["complete"] is True
  assert ep.stats["page_size_reduced_to"] <= 40
  pd.testing.assert_frame_equal(canon(df), canon(file_frame(graph)))


def test_gives_up_with_clear_error_at_minimum_page_size(graph):
  url, stop, _ = serve(graph, break_limit_above=0)   # every page is cut off
  try:
    ep = gl.SparqlEndpoint(url, page_size=40, min_page_size=20, retries=1, backoff=0)
    with pytest.raises(gl.TransientEndpointError):
      ep.select(QUERY)
  finally:
    stop()


PART_QUERY = QUERY.replace("OPTIONAL { ?e ex:rate ?rate }   # rate is optional",
                           "OPTIONAL { ?e ex:rate ?rate }\n  # GRAPHLENS-PARTITION ?e")


@pytest.mark.parametrize("parts", [16, 256])
def test_partitioned_query_returns_everything_once(graph, parts):
  url, stop, state = serve(graph, row_cap=20)
  try:
    ep = gl.SparqlEndpoint(url, page_size=50, partitions=parts, backoff=0)
    df = ep.select(PART_QUERY)
  finally:
    stop()
  pd.testing.assert_frame_equal(canon(df), canon(file_frame(graph)))
  assert ep.stats["partitions"] == parts and ep.stats["complete"] is True
  assert ep.stats["expected_rows"] == 137 and ep.stats["duplicate_rows"] == 0
  assert sum("MD5(STR(?e))" in q for q in state["queries"]) >= parts


def test_partition_marker_is_a_plain_comment_elsewhere(graph):
  # file mode (rdflib) and --partitions 1 both ignore the marker
  pd.testing.assert_frame_equal(canon(file_frame(graph, PART_QUERY)), canon(file_frame(graph)))
  url, stop, state = serve(graph)
  try:
    ep = gl.SparqlEndpoint(url, page_size=500, partitions=1, backoff=0)
    assert len(ep.select(PART_QUERY)) == 137
  finally:
    stop()
  assert not any("MD5" in q for q in state["queries"])


def test_group_by_timeout_error_suggests_partitioning(graph):
  q = """PREFIX ex: <http://ex.org/>
SELECT ?e (MAX(?submitted) AS ?s) WHERE { ?e ex:submitted ?submitted } GROUP BY ?e"""
  url, stop, _ = serve(graph, partial_first=10 ** 6)        # every page times out
  try:
    ep = gl.SparqlEndpoint(url, page_size=20, min_page_size=20, retries=1, backoff=0)
    with pytest.raises(gl.TransientEndpointError, match="GRAPHLENS-PARTITION"):
      ep.select(q)
  finally:
    stop()


def test_max_rows_with_partitions_is_a_whole_slice_sample(graph):
  """--max-rows on a partitioned query fetches whole hash slices until the limit is reached:
  a reproducible random sample, the same in every run."""
  url, stop, _ = serve(graph, row_cap=20)
  try:
    runs = []
    for _ in range(2):
      ep = gl.SparqlEndpoint(url, page_size=50, partitions=16, max_rows=40, backoff=0)
      runs.append(canon(ep.select(PART_QUERY)))
  finally:
    stop()
  df = runs[0]
  pd.testing.assert_frame_equal(df, runs[1])
  k = ep.stats["sample_partitions"]
  assert 40 <= len(df) < 137 and 0 < k < 16
  full = file_frame(graph)
  import hashlib
  want = full[full["e"].map(lambda e: int(hashlib.md5(str(e).encode()).hexdigest()[0], 16) < k)]
  assert len(df) == len(want) and ep.stats["complete"] is True
