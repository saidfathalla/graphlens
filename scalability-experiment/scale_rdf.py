#!/usr/bin/env python3
"""Create a synthetic RDF workload by cloning target-class instances' outgoing triples.

This is a controlled processing-load generator, not a substitute for naturally
occurring RDF graphs. Cloned entities reuse original objects/categories and are
suitable only when the SPARQL query does not filter for a fixed URI prefix/list.
"""
import argparse
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDF
from rdflib.util import guess_format

ap = argparse.ArgumentParser()
ap.add_argument('--input', required=True, help='Input RDF file')
ap.add_argument('--output', required=True, help='Output RDF file')
ap.add_argument('--target-class', required=True, help='Full URI of the entity class selected by the SPARQL query')
ap.add_argument('--target-triples', type=int, required=True, help='Minimum desired triple count')
args = ap.parse_args()

fmt_in = guess_format(args.input)
fmt_out = guess_format(args.output) or fmt_in or 'turtle'
g = Graph()
g.parse(args.input, format=fmt_in)
initial = len(g)
if args.target_triples <= initial:
    g.serialize(destination=args.output, format=fmt_out)
    print(f'Input already has {initial} triples; wrote {len(g)} triples to {args.output}')
    raise SystemExit(0)

cls = URIRef(args.target_class)
subjects = sorted({s for s in g.subjects(RDF.type, cls) if isinstance(s, URIRef)}, key=str)
if not subjects:
    raise SystemExit(f'No URI subjects typed as {args.target_class}; check the class URI and RDF data.')
records = [(s, list(g.predicate_objects(s))) for s in subjects]
records = [(s, triples) for s, triples in records if triples]
if not records:
    raise SystemExit('Target-class subjects have no outgoing triples.')

added = 0
clone_no = 0
while len(g) < args.target_triples:
    source, triples = records[clone_no % len(records)]
    clone_no += 1
    clone = URIRef(f'{str(source).rstrip("/#")}/graphlens-scale-{clone_no}')
    # Avoid URI collisions with any existing subject/object in the source graph.
    while (clone, None, None) in g or (None, None, clone) in g:
        clone_no += 1
        clone = URIRef(f'{str(source).rstrip("/#")}/graphlens-scale-{clone_no}')
    before = len(g)
    for p, o in triples:
        g.add((clone, p, o))
    added += len(g) - before

g.serialize(destination=args.output, format=fmt_out)
print(f'Input triples: {initial}')
print(f'Target triples: {args.target_triples}')
print(f'Actual triples written: {len(g)}')
print(f'Target-class source entities: {len(records)}')
print(f'Cloned records: {clone_no}')
print(f'Added triples: {added}')
print(f'Output: {args.output}')
