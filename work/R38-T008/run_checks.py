#!/usr/bin/env python3
"""
Verification check for R38-T008.
Saves inputs, expected vs actual outputs, tool versions, and exits non-zero on assertion failure.
"""

import sys
import os
import json
import platform

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../lessons/4-ComputerVision/09-Autoencoders/applied-labs/r38-t008")))
from example import AutoencoderBatchPreprocessor

def main():
    print("Running verification for R38-T008...")
    work_dir = os.path.dirname(os.path.abspath(__file__))

    tool_versions = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "task_id": "R38-T008",
        "lesson": "lessons/4-ComputerVision/09-Autoencoders"
    }
    with open(os.path.join(work_dir, "tool_versions.txt"), "w") as f:
        for k, v in tool_versions.items():
            f.write(f"{k}: {v}\n")

    fixtures = {
        "valid_batch": [[[0, 255], [128, 64]]],
        "invalid_batch": [[[0, -50], [128, 64]]]
    }
    with open(os.path.join(work_dir, "inputs.json"), "w") as f:
        json.dump(fixtures, f, indent=2)

    processor = AutoencoderBatchPreprocessor()
    
    # Run failure scenario
    err_res = processor.safe_process_with_recovery(fixtures["invalid_batch"])
    assert err_res["success"] is False, "Invalid batch should have failed"

    # Run recovery scenario
    rec_res = processor.safe_process_with_recovery(fixtures["valid_batch"])
    assert rec_res["success"] is True, "Preprocessor failed to recover on valid batch"

    expected = {
        "shape": [1, 2, 2],
        "normalized": [[[0.0, 1.0], [128 / 255.0, 64 / 255.0]]]
    }
    actual = {
        "shape": list(rec_res["shape"]),
        "normalized": rec_res["data"]
    }

    with open(os.path.join(work_dir, "results.json"), "w") as f:
        json.dump({"expected": expected, "actual": actual}, f, indent=2)

    assert actual["shape"] == expected["shape"], "Shape mismatch"
    assert abs(actual["normalized"][0][0][1] - 1.0) < 1e-6, "Pixel 255 must map to 1.0"

    print("All checks passed successfully for R38-T008.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
