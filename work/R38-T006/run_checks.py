#!/usr/bin/env python3
"""
Verification check for R38-T006.
Saves inputs, expected vs actual outputs, tool versions, and exits non-zero on assertion failure.
"""

import sys
import os
import json
import platform

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../lessons/6-Other/23-MultiagentSystems/applied-labs/r38-t006")))
from example import baseline_validate_and_order, refactored_validate_and_order, MissingDependencyError, CyclicDependencyError

def main():
    print("Running verification for R38-T006...")
    work_dir = os.path.dirname(os.path.abspath(__file__))

    tool_versions = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "task_id": "R38-T006",
        "lesson": "lessons/6-Other/23-MultiagentSystems"
    }
    with open(os.path.join(work_dir, "tool_versions.txt"), "w") as f:
        for k, v in tool_versions.items():
            f.write(f"{k}: {v}\n")

    input_fixture = {
        "A": [],
        "B": ["A"],
        "C": ["B"]
    }
    with open(os.path.join(work_dir, "inputs.json"), "w") as f:
        json.dump({"canonical_plan": input_fixture}, f, indent=2)

    actual_baseline = baseline_validate_and_order(input_fixture)
    actual_refactored = refactored_validate_and_order(input_fixture)
    expected_order = ["A", "B", "C"]

    with open(os.path.join(work_dir, "results.json"), "w") as f:
        json.dump({
            "expected": expected_order,
            "actual_baseline": actual_baseline,
            "actual_refactored": actual_refactored,
            "equivalent": actual_baseline == actual_refactored
        }, f, indent=2)

    assert actual_refactored == expected_order, f"Expected {expected_order}, got {actual_refactored}"
    assert actual_baseline == actual_refactored, "Refactored implementation diverged from baseline"

    # Verify cycle rejection
    try:
        refactored_validate_and_order({"A": ["B"], "B": ["A"]})
        raise AssertionError("Cycle was not rejected")
    except CyclicDependencyError:
        pass

    # Verify missing dependency reporting
    try:
        refactored_validate_and_order({"A": ["MISSING_DEP"]})
        raise AssertionError("Missing dependency was not rejected")
    except MissingDependencyError as e:
        assert "MISSING_DEP" in str(e)

    print("All checks passed successfully for R38-T006.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
