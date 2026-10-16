# GraphLens analytical-correctness test dataset

## Purpose

This is a small, synthetic RDF dataset designed to test GraphLens normalization and deterministic recovery against manually calculated expected values. It is a reproducibility/unit-test experiment, not evidence about real educational programs.

## Files

- `analytical_correctness.ttl`: 12 synthetic training programs (92 RDF triples).
- `analytical_correctness.rq`: SPARQL projection with optional completion count and completion rate.
- `profile.yaml`: maps the RDF vocabulary to GraphLens analytical roles.
- `expected_manual_results.csv`: independently specified expected normalized values and provenance flags.
- `validate_results.py`: compares the GraphLens feature matrix with expected values and checks missingness and provenance flags.
- `output/features_matrix.csv`: output from the GraphLens run used for this package.
- `output/run_report.json`: GraphLens execution report.

## Reproduce

From this directory, run:

```bash
python3 /path/to/graphlens.py -f analytical_correctness.ttl -q analytical_correctness.rq -p profile.yaml -o output --log-level INFO
python3 validate_results.py
```

The GraphLens source path may be changed to the local path of your checked-out version. The bundled run was executed with seed 42 and the GraphLens source available in the working environment.

## Manual calculations and test cases

- P01: rate 0.8 is already a ratio; expected completed count = 80.
- P02: rate 75 is interpreted as a percentage, so 75/100 = 0.75.
- P03: missing rate is recovered as 99/110 = 0.9.
- P04: missing completed count is recovered as round(80*0.625) = 50.
- P05: completed count 120 exceeds enrolled count 100 and is flagged inconsistent. Given rate 0.8, completed count is reconstructed as round(100*0.8) = 80; the inconsistency flag remains for audit.
- P06: rate 150 is rescaled to 1.5 and then rejected as outside [0,1]. Completed count is also missing, so neither value can be recovered.
- P07: rate -0.2 is rejected as outside [0,1]. Because completed and enrolled counts are available, the rate is recovered as 50/100 = 0.5.
- P08: rate is missing and enrolled count is zero. A rate cannot be derived by division by zero; the rate remains unresolved.
- P09: rate 0.7 agrees with 210/300.
- P10: rate 0.4 agrees with 20/50.
- P11: rate 50 is interpreted as 50%, giving 0.5.
- P12: rate 0.5 agrees with 80/160.

## Results from the bundled run

- RDF triples: 92.
- SPARQL rows / final feature rows: 12 / 12.
- Percentage-scale values rescaled: 3 (including P06, which is subsequently rejected as invalid).
- Completion rates recovered: 2 (P03 and P07).
- Completed counts recovered: 2 (P04 and P05).
- Inconsistent records flagged: 1 (P05).
- Final missing values: one `x2` and two `y` values.
- Rows with at least one unresolved `x2` or `y`: 2 (P06 and P08).
- Compared numerical values: 11 `x2` values and 10 `y` values.
- Maximum absolute error: 0 for `x2` and `y`.
- Mean absolute error: 0 for `x2` and `y`.
- Validation script status: PASS.

## Interpretation and limitations

The results show agreement between the manually specified expected outputs and this particular GraphLens run for the tested cases. The test covers percentage conversion, recoverable missing values, impossible counts, out-of-range ratios, and a zero-denominator case. It does not validate every possible malformed RDF value, nor does it establish general correctness for arbitrary graphs. The test also exposes implementation behavior that should be described accurately: values greater than 1 are first treated as percentages; values still outside [0,1] after rescaling are set missing; deterministic recovery is then attempted from the other available count/rate. An inconsistency flag remains set even if the impossible count is reconstructed.
