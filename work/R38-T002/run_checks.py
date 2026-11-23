#!/usr/bin/env python3
"""
Verification check for R38-T002.
Saves inputs, expected vs actual outputs, tool versions, and exits non-zero on assertion failure.
"""

import sys
import os
import json
import platform

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../lessons/X-Extras/X1-MultiModal/applied-labs/r38-t002")))
from example import evaluate_binary_classification

def main():
    print("Running verification for R38-T002...")
    work_dir = os.path.dirname(os.path.abspath(__file__))

    tool_versions = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "task_id": "R38-T002",
        "lesson": "lessons/X-Extras/X1-MultiModal"
    }
    with open(os.path.join(work_dir, "tool_versions.txt"), "w") as f:
        for k, v in tool_versions.items():
            f.write(f"{k}: {v}\n")

    input_fixture = {
        "y_true": [1, 1, 0, 0],
        "y_pred": [1, 0, 1, 0]
    }
    with open(os.path.join(work_dir, "inputs.json"), "w") as f:
        json.dump(input_fixture, f, indent=2)

    actual = evaluate_binary_classification(input_fixture["y_true"], input_fixture["y_pred"])
    expected = {
        "TP": 1,
        "FN": 1,
        "FP": 1,
        "TN": 1,
        "precision": 0.5,
        "recall": 0.5
    }

    with open(os.path.join(work_dir, "results.json"), "w") as f:
        json.dump({"expected": expected, "actual": actual}, f, indent=2)

    # Check contract assertions
    assert actual == expected, f"Mismatch: {actual} vs {expected}"

    # Zero denominator check
    zero_pred_res = evaluate_binary_classification([1, 1, 0, 0], [0, 0, 0, 0])
    assert zero_pred_res["precision"] == 0.0, "Zero denominator must yield precision=0.0"

    print("All checks passed successfully for R38-T002.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
