#!/usr/bin/env python3
"""Compare GraphLens output against independently specified expected values."""
import math
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent
actual = pd.read_csv(root / "output" / "features_matrix.csv")
expected = pd.read_csv(root / "expected_manual_results.csv")

merged = expected.merge(actual, left_on="programCode", right_on="e", how="outer", indicator=True)
assert (merged["_merge"] == "both").all(), "Program IDs differ between expected and actual outputs"

numeric_errors = {"x2": [], "y": []}
flag_checks = []
unresolved_count = 0
for _, row in merged.iterrows():
    for field, expected_field in (("x2", "expected_x2"), ("y", "expected_y")):
        exp = row[expected_field]
        got = row[field]
        exp_missing = pd.isna(exp)
        got_missing = pd.isna(got)
        assert exp_missing == got_missing, f"{row['programCode']}: missingness mismatch for {field}: expected={exp}, actual={got}"
        if not exp_missing:
            err = abs(float(got) - float(exp))
            numeric_errors[field].append(err)
            assert math.isclose(float(got), float(exp), rel_tol=0, abs_tol=1e-12), (
                f"{row['programCode']}: {field} mismatch expected={exp}, actual={got}"
            )
    for actual_field, expected_field in (
        ("y_rescaled", "expected_y_rescaled"),
        ("y_recovered", "expected_y_recovered"),
        ("x2_recovered", "expected_x2_recovered"),
        ("inconsistent", "expected_inconsistent"),
    ):
        exp = str(row[expected_field]).lower() == "true"
        got = bool(row[actual_field])
        assert got == exp, f"{row['programCode']}: {actual_field} mismatch expected={exp}, actual={got}"
    unresolved = pd.isna(row["x2"]) or pd.isna(row["y"])
    assert unresolved == (str(row["expected_unresolved"]).lower() == "true"), (
        f"{row['programCode']}: unresolved flag mismatch"
    )
    unresolved_count += int(unresolved)

report = {
    "records_expected": len(expected),
    "records_actual": len(actual),
    "numeric_fields_compared": {k: len(v) for k, v in numeric_errors.items()},
    "max_absolute_error": {k: max(v) if v else None for k, v in numeric_errors.items()},
    "mean_absolute_error": {k: sum(v) / len(v) if v else None for k, v in numeric_errors.items()},
    "y_rescaled": int(actual["y_rescaled"].sum()),
    "y_recovered": int(actual["y_recovered"].sum()),
    "x2_recovered": int(actual["x2_recovered"].sum()),
    "inconsistent": int(actual["inconsistent"].sum()),
    "records_with_unresolved_x2_or_y": unresolved_count,
    "status": "PASS",
}
print(pd.Series(report).to_string())
