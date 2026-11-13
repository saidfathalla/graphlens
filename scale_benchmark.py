#!/usr/bin/env python3
"""Measure (not extrapolate) GraphLens time and peak RSS on replicated EVENTSKG graphs.
Each replica gets fresh IRIs (suffix _cpN) so triples are distinct; replicas keep the original value
distributions, so row counts scale with triples. UNTESTED in the authoring sandbox (no rdflib there): run it once and inspect.
Usage: python scale_benchmark.py -f eventskg.ttl -q query.rq -p profile.yaml -t training_data.csv --replicas 1 4 16 64
"""
import argparse, json, subprocess, sys, time, tempfile, os
import psutil, rdflib

def replicate(src, n, ns, dst):
    g = rdflib.Graph(); g.parse(src, format="turtle")
    out = rdflib.Graph()
    def tag(t, i):
        return rdflib.URIRef(str(t) + f"_cp{i}") if isinstance(t, rdflib.URIRef) and str(t).startswith(ns) else t
    for i in range(n):
        for s, p, o in g:
            out.add((tag(s, i), p, tag(o, i)))
    out.serialize(dst, format="turtle"); return len(out)

def run(cmd):
    t0 = time.perf_counter(); p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ps = psutil.Process(p.pid); peak = 0
    while p.poll() is None:
        try: peak = max(peak, ps.memory_info().rss + sum(c.memory_info().rss for c in ps.children(recursive=True)))
        except psutil.Error: pass
        time.sleep(0.05)
    return p.returncode, time.perf_counter() - t0, peak / 2**20

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", required=True); ap.add_argument("-q", required=True); ap.add_argument("-p", required=True)
    ap.add_argument("-t", default="training_data.csv"); ap.add_argument("--ns", default="http://w3id.org/EVENTSKG-Dataset/ekg#")
    ap.add_argument("--replicas", type=int, nargs="+", default=[1, 4, 16, 64]); ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args(); results = []
    for n in a.replicas:
        with tempfile.TemporaryDirectory() as d:
            ttl = os.path.join(d, "g.ttl"); triples = replicate(a.f, n, a.ns, ttl)
            for r in range(a.repeats):
                rc, sec, mb = run([sys.executable, "graphlens.py", "-q", a.q, "-p", a.p, "-t", a.t, "-f", ttl, "-o", os.path.join(d, "o")])
                results.append(dict(replicas=n, triples=triples, run=r, rc=rc, seconds=round(sec, 2), peak_rss_mb=round(mb, 1)))
                print(results[-1], flush=True)
    json.dump(results, open("scale_benchmark.json", "w"), indent=1)
