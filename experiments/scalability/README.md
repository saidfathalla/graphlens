# GraphLens scalability benchmark helpers

`benchmark.sh` runs one warm-up plus repeated measured GraphLens executions. It uses GNU `/usr/bin/time` to capture wall-clock time and peak RSS for the complete child process, including RDF loading, SPARQL, analytics, and chart rendering. Results are saved to CSV.

`scale_rdf.py` can produce approximate larger synthetic RDF workloads by cloning outgoing triples of entities typed with a chosen class. This is a controlled workload generator, not a representative replacement for a naturally grown knowledge graph. Verify that the SPARQL query does not constrain entity URIs to a fixed list or prefix before using it.

Example, from the repository root (the enlarged graphs are not tracked; create them first):

```bash
python3 experiments/scalability/scale_rdf.py --input data/eventskg.ttl \
    --output experiments/scalability/EVENTSKG_500k.ttl --target-triples 500000 \
    --target-class "https://w3id.org/scholarlydata/ontology/conference-ontology.owl#Conference"
bash experiments/scalability/benchmark.sh graphlens.py experiments/scalability/EVENTSKG_500k.ttl \
    queries/eventskg-query.rq profiles/eventskg-profile.yaml experiments/scalability/results_500k 5
```

`results_original_timings.csv`, `results_100k_timings.csv` and `results_500k_timings.csv` hold the measured runs (Intel Core Ultra 5 225U, 30 GB RAM).
