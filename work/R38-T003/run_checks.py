#!/usr/bin/env python3
"""
Verification check for R38-T003.
Saves inputs, expected vs actual outputs, tool versions, and exits non-zero on assertion failure.
"""

import sys
import os
import json
import platform

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../lessons/3-NeuralNetworks/04-OwnFramework/applied-labs/r38-t003")))
from example import direct_perceptron_step, independent_perceptron_step, compare_approaches

def main():
    print("Running verification for R38-T003...")
    work_dir = os.path.dirname(os.path.abspath(__file__))

    tool_versions = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "task_id": "R38-T003",
        "lesson": "lessons/3-NeuralNetworks/04-OwnFramework"
    }
    with open(os.path.join(work_dir, "tool_versions.txt"), "w") as f:
        for k, v in tool_versions.items():
            f.write(f"{k}: {v}\n")

    input_fixture = {
        "weights": [0.0, 0.0],
        "bias": 0.0,
        "x": [1.0, 2.0],
        "target": 1,
        "learning_rate": 1.0
    }
    with open(os.path.join(work_dir, "inputs.json"), "w") as f:
        json.dump(input_fixture, f, indent=2)

    actual = direct_perceptron_step(
        input_fixture["weights"],
        input_fixture["bias"],
        input_fixture["x"],
        input_fixture["target"],
        input_fixture["learning_rate"]
    )
    expected = {
        "score": 0.0,
        "initial_prediction": 0,
        "error": 1,
        "weights": [1.0, 2.0],
        "bias": 1.0
    }

    with open(os.path.join(work_dir, "results.json"), "w") as f:
        json.dump({"expected": expected, "actual": actual}, f, indent=2)

    # Check contract assertions
    assert actual["initial_prediction"] == 0, f"Expected initial prediction 0, got {actual['initial_prediction']}"
    assert actual["weights"] == [1.0, 2.0], f"Expected weights [1.0, 2.0], got {actual['weights']}"
    assert actual["bias"] == 1.0, f"Expected bias 1.0, got {actual['bias']}"

    # Comparison check
    comp = compare_approaches([0.0, 0.0], 0.0, [1.0, 2.0], 1, 1.0)
    assert comp["equivalent"] is True, "Approaches failed equivalence check"

    print("All checks passed successfully for R38-T003.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
