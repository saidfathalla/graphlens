"""Tiny SPARQL 1.1 Protocol server over an rdflib Graph, for offline tests.

Can mimic real endpoints: a per-request row cap (like Virtuoso's MaxResultRows),
a paging-depth limit (rows beyond position `window_cap` are never returned, as DBpedia
does for sorted queries), Virtuoso's time-limit behaviour (an empty page flagged with the
partial-result header for the first `partial_first` pages; a flagged COUNT with
`partial_count`), initial 503 failures, and
responses cut off mid-transfer (the first `break_first` pages, and every page whose LIMIT
exceeds `break_limit_above`).
"""
import re
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def serve(graph, row_cap=None, fail_first=0, window_cap=None, partial_first=0,
          partial_count=False, break_first=0, break_limit_above=None):
  state = {"fail": fail_first, "break": break_first, "partial": partial_first, "queries": []}

  class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
      pass

    def _answer(self, query):
      if state["fail"] > 0:
        state["fail"] -= 1
        self.send_response(503); self.send_header("Retry-After", "0"); self.end_headers()
        return
      state["queries"].append(query)
      res = graph.query(query)
      if row_cap is not None:            # emulate a server-side cap on rows per response
        res.bindings = res.bindings[:row_cap]
      is_count = "COUNT(*)" in query
      partial = (partial_count and is_count) or (not is_count and state["partial"] > 0)
      if partial and not is_count:       # Virtuoso: time limit hit, nothing (or little) returned
        state["partial"] -= 1
        res.bindings = []
      m = re.search(r"OFFSET\s+(\d+)\s*$", query)
      if window_cap is not None and m:   # emulate a paging-depth limit
        res.bindings = res.bindings[: max(0, window_cap - int(m.group(1)))]
      body = res.serialize(format="json")
      lim = re.search(r"LIMIT\s+(\d+)", query)
      cut = "COUNT" not in query and (state["break"] > 0 or (
          break_limit_above is not None and lim and int(lim.group(1)) > break_limit_above))
      if cut:
        state["break"] = max(0, state["break"] - 1)
        self.send_response(200)
        self.send_header("Content-Type", "application/sparql-results+json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body[: len(body) // 3])   # then drop the connection
        self.close_connection = True
        return
      self.send_response(200)
      if partial:
        self.send_header("X-SQL-State", "S1TAT")
        self.send_header("X-SQL-Message", "RC...: Returning incomplete results")
      self.send_header("Content-Type", "application/sparql-results+json")
      self.send_header("Content-Length", str(len(body)))
      self.end_headers()
      self.wfile.write(body)

    def do_POST(self):
      n = int(self.headers.get("Content-Length", 0))
      self._answer(urllib.parse.parse_qs(self.rfile.read(n).decode())["query"][0])

    def do_GET(self):
      self._answer(urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)["query"][0])

  srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
  threading.Thread(target=srv.serve_forever, daemon=True).start()
  return f"http://127.0.0.1:{srv.server_address[1]}/sparql", srv.shutdown, state
