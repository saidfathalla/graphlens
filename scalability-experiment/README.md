# GraphLens scalability benchmark helpers

`benchmark.sh` runs one warm-up plus repeated measured GraphLens executions. It uses GNU `/usr/bin/time` to capture wall-clock time and peak RSS for the complete child process, including RDF loading, SPARQL, analytics, and chart rendering. Results are saved to CSV.

`scale_rdf.py` can produce approximate larger synthetic RDF workloads by cloning outgoing triples of entities typed with a chosen class. This is a controlled workload generator, not a representative replacement for a naturally grown knowledge graph. Verify that the SPARQL query does not constrain entity URIs to a fixed list or prefix before using it.

Example:

```bash
./benchmark.sh /path/to/graphlens.py /path/to/data.ttl /path/to/query.rq /path/to/profile.yaml results_27581 5
```
