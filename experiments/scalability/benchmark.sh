#!/usr/bin/env bash
set -euo pipefail

# Usage: ./benchmark.sh graphlens.py dataset.ttl query.rq profile.yaml output_prefix [measured_runs]
GRAPH_CODE="${1:?GraphLens Python file required}"
RDF_FILE="${2:?RDF dataset required}"
QUERY_FILE="${3:?SPARQL query required}"
PROFILE_FILE="${4:?YAML profile required}"
PREFIX="${5:?Output prefix required}"
RUNS="${6:-5}"

for f in "$GRAPH_CODE" "$RDF_FILE" "$QUERY_FILE" "$PROFILE_FILE"; do
  [[ -f "$f" ]] || { echo "File not found: $f" >&2; exit 2; }
done
command -v /usr/bin/time >/dev/null || { echo "GNU /usr/bin/time is required" >&2; exit 2; }

TRIPLES=$(python3 - "$RDF_FILE" <<'PY'
import sys
from rdflib import Graph
from rdflib.util import guess_format
p = sys.argv[1]
g = Graph()
g.parse(p, format=guess_format(p))
print(len(g))
PY
)
CSV="${PREFIX}_timings.csv"
printf 'run,triples,elapsed_seconds,max_rss_kib,status\n' > "$CSV"

run_one() {
  local label="$1"
  local out="${PREFIX}_${label}_output"
  local timing="${PREFIX}_${label}_time.txt"
  rm -rf "$out"
  mkdir -p "$out"
  local status=0
  /usr/bin/time -f '%e,%M' -o "$timing" \
    python3 "$GRAPH_CODE" -f "$RDF_FILE" -q "$QUERY_FILE" -p "$PROFILE_FILE" \
      -o "$out" --log-level WARNING || status=$?
  if [[ -s "$timing" ]]; then
    local vals
    vals=$(cat "$timing")
    printf '%s,%s,%s,%s,%s\n' "$label" "$TRIPLES" "${vals%,*}" "${vals#*,}" "$status" >> "$CSV"
  else
    printf '%s,%s,,,failed\n' "$label" "$TRIPLES" >> "$CSV"
  fi
  if [[ "$status" -ne 0 ]]; then
    echo "Run $label failed (exit $status); see terminal output." >&2
    return "$status"
  fi
}

# One unreported warm-up run to reduce first-run/import effects.
run_one warmup
for i in $(seq 1 "$RUNS"); do
  run_one "$i"
done

python3 - "$CSV" <<'PY'
import csv, statistics, sys
p=sys.argv[1]
with open(p, newline='') as f:
    rows=[r for r in csv.DictReader(f) if r['run']!='warmup' and r['status']=='0']
if not rows:
    raise SystemExit('No successful measured runs found.')
times=[float(r['elapsed_seconds']) for r in rows]
mem=[int(r['max_rss_kib'])/1024 for r in rows]
print(f'CSV: {p}')
print(f'Successful measured runs: {len(rows)}')
print(f'Median elapsed time: {statistics.median(times):.3f} s')
print(f'Min–max elapsed time: {min(times):.3f}–{max(times):.3f} s')
print(f'Median peak RSS: {statistics.median(mem):.1f} MiB')
print(f'Maximum observed peak RSS: {max(mem):.1f} MiB')
PY
