#!/usr/bin/env python3
"""
Verification check for R38-T010.
Saves inputs, expected vs actual outputs, tool versions, and exits non-zero on assertion failure.
"""

import sys
import os
import json
import platform

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../lessons/4-ComputerVision/11-ObjectDetection/applied-labs/r38-t010")))
from example import compute_iou, scenario_assessment

def main():
    print("Running verification for R38-T010...")
    work_dir = os.path.dirname(os.path.abspath(__file__))

    tool_versions = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "task_id": "R38-T010",
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

    actual_iou = compute_iou(input_fixture["box_a"], input_fixture["box_b"])
    assessment = scenario_assessment()

    expected = {
        "iou": 1.0 / 7.0,
        "q1_expected": 1.0 / 7.0,
        "q3_expected": 0.0
    }
    actual = {
        "iou": actual_iou,
        "q1_answer": assessment["q1"]["answer"],
        "q3_answer": assessment["q3"]["answer"],
        "assessment": assessment
    }

    with open(os.path.join(work_dir, "results.json"), "w") as f:
        json.dump({"expected": expected, "actual": actual}, f, indent=2)

    assert abs(actual_iou - (1.0 / 7.0)) < 1e-6, f"Expected IoU 1/7, got {actual_iou}"
    assert abs(assessment["q1"]["answer"] - (1.0 / 7.0)) < 1e-6, "Q1 prediction mismatch"
    assert assessment["q3"]["answer"] == 0.0, "Q3 boundary policy mismatch"

    print("All checks passed successfully for R38-T010.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
