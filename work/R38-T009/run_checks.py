#!/usr/bin/env python3
"""
Verification check for R38-T009.
Saves inputs, expected vs actual outputs, tool versions, and exits non-zero on assertion failure.
"""

import sys
import os
import json
import platform

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../lessons/3-NeuralNetworks/03-Perceptron/applied-labs/r38-t009")))
from example import perceptron_update_step, run_configuration_experiment

def main():
    print("Running verification for R38-T009...")
    work_dir = os.path.dirname(os.path.abspath(__file__))

    tool_versions = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "task_id": "R38-T009",
        "lesson": "lessons/3-NeuralNetworks/03-Perceptron"
    }
    with open(os.path.join(work_dir, "tool_versions.txt"), "w") as f:
        for k, v in tool_versions.items():
            f.write(f"{k}: {v}\n")

    input_fixture = {
        "weights": [0.0, 0.0],
        "bias": 0.0,
        "x": [1.0, 2.0],
        "target": 1,
        "learning_rates": [0.1, 0.5, 1.0]
    }
    with open(os.path.join(work_dir, "inputs.json"), "w") as f:
        json.dump(input_fixture, f, indent=2)

    exp_res = run_configuration_experiment(
        input_fixture["weights"],
        input_fixture["bias"],
        input_fixture["x"],
        input_fixture["target"],
        input_fixture["learning_rates"]
    )

    expected_canonical = {
        "weights": [1.0, 2.0],
        "bias": 1.0
    }
    actual_canonical = {
        "weights": exp_res["configurations"]["lr_1.0"]["updated_weights"],
        "bias": exp_res["configurations"]["lr_1.0"]["updated_bias"]
    }

    with open(os.path.join(work_dir, "results.json"), "w") as f:
        json.dump({
            "expected_canonical": expected_canonical,
            "actual_canonical": actual_canonical,
            "experiment_summary": exp_res
        }, f, indent=2)

    assert actual_canonical["weights"] == expected_canonical["weights"], "Canonical weights mismatch"
    assert actual_canonical["bias"] == expected_canonical["bias"], "Canonical bias mismatch"
    assert len(exp_res["configurations"]) == 3, "Must evaluate 3 configurations"

    print("All checks passed successfully for R38-T009.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
