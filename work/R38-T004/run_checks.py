#!/usr/bin/env python3
"""
Verification check for R38-T004.
Saves inputs, expected vs actual outputs, tool versions, and exits non-zero on assertion failure.
"""

import sys
import os
import json
import platform

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../lessons/4-ComputerVision/11-ObjectDetection/applied-labs/r38-t004")))
from example import compute_iou_with_trace

def main():
    print("Running verification for R38-T004...")
    work_dir = os.path.dirname(os.path.abspath(__file__))

    tool_versions = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "task_id": "R38-T004",
        "lesson": "lessons/4-ComputerVision/11-ObjectDetection"
    }
    with open(os.path.join(work_dir, "tool_versions.txt"), "w") as f:
        for k, v in tool_versions.items():
            f.write(f"{k}: {v}\n")

    input_fixture = {
        "box_a": [0, 0, 2, 2],
        "box_b": [1, 1, 3, 3]
    }
    with open(os.path.join(work_dir, "inputs.json"), "w") as f:
        json.dump(input_fixture, f, indent=2)

    actual = compute_iou_with_trace(input_fixture["box_a"], input_fixture["box_b"])
    expected = {
        "area_a": 4.0,
        "area_b": 4.0,
        "intersection_area": 1.0,
        "union_area": 7.0,
        "iou": 1.0 / 7.0,
        "invariant_holds": True
    }

    with open(os.path.join(work_dir, "results.json"), "w") as f:
        json.dump({"expected": expected, "actual": actual}, f, indent=2)

    # Check contract assertions
    assert abs(actual["iou"] - (1.0 / 7.0)) < 1e-6, f"Expected IoU 1/7, got {actual['iou']}"
    assert actual["invariant_holds"] is True, "State invariant must hold"

    # Disjoint check
    disjoint_res = compute_iou_with_trace([0, 0, 2, 2], [10, 10, 12, 12])
    assert disjoint_res["iou"] == 0.0, "Disjoint boxes must have IoU = 0"

    # Identical check
    identical_res = compute_iou_with_trace([0, 0, 2, 2], [0, 0, 2, 2])
    assert abs(identical_res["iou"] - 1.0) < 1e-6, "Identical boxes must have IoU = 1"

    print("All checks passed successfully for R38-T004.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
