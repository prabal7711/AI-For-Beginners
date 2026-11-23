#!/usr/bin/env python3
"""
Verification check for R38-T007.
Saves inputs, expected vs actual outputs, tool versions, and exits non-zero on assertion failure.
"""

import sys
import os
import json
import platform

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../lessons/4-ComputerVision/08-TransferLearning/applied-labs/r38-t007")))
from example import MeanImputer

def main():
    print("Running verification for R38-T007...")
    work_dir = os.path.dirname(os.path.abspath(__file__))

    tool_versions = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "task_id": "R38-T007",
        "lesson": "lessons/4-ComputerVision/08-TransferLearning"
    }
    with open(os.path.join(work_dir, "tool_versions.txt"), "w") as f:
        for k, v in tool_versions.items():
            f.write(f"{k}: {v}\n")

    input_fixture = {
        "train": [2, None, 4],
        "validation": [100, None]
    }
    with open(os.path.join(work_dir, "inputs.json"), "w") as f:
        json.dump(input_fixture, f, indent=2)

    imputer = MeanImputer()
    imputer.fit(input_fixture["train"])
    actual_val = imputer.transform(input_fixture["validation"])

    expected = {
        "fitted_mean": 3.0,
        "imputed_validation": [100.0, 3.0]
    }
    actual = {
        "fitted_mean": imputer.fitted_mean,
        "imputed_validation": actual_val
    }

    with open(os.path.join(work_dir, "results.json"), "w") as f:
        json.dump({"expected": expected, "actual": actual}, f, indent=2)

    assert actual["fitted_mean"] == expected["fitted_mean"], f"Expected mean 3.0, got {actual['fitted_mean']}"
    assert actual["imputed_validation"] == expected["imputed_validation"], f"Expected [100.0, 3.0], got {actual['imputed_validation']}"

    # Verify all-missing training input is rejected
    try:
        MeanImputer().fit([None, None])
        raise AssertionError("All-missing training input was not rejected")
    except ValueError:
        pass

    print("All checks passed successfully for R38-T007.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
