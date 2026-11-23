#!/usr/bin/env python3
"""
Verification check for R38-T005.
Saves inputs, expected vs actual outputs, tool versions, and exits non-zero on assertion failure.
"""

import sys
import os
import json
import platform

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../lessons/3-NeuralNetworks/05-Frameworks/applied-labs/r38-t005")))
from example import ComponentRouter

def main():
    print("Running verification for R38-T005...")
    work_dir = os.path.dirname(os.path.abspath(__file__))

    tool_versions = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "task_id": "R38-T005",
        "lesson": "lessons/3-NeuralNetworks/05-Frameworks"
    }
    with open(os.path.join(work_dir, "tool_versions.txt"), "w") as f:
        for k, v in tool_versions.items():
            f.write(f"{k}: {v}\n")

    input_fixture = [
        "Need help with setup",
        "calculate batch gradient",
        "Help calculate cross-entropy",
        "play background music"
    ]
    with open(os.path.join(work_dir, "inputs.json"), "w") as f:
        json.dump({"test_requests": input_fixture}, f, indent=2)

    router = ComponentRouter()
    actual_results = [router.route_request(req) for req in input_fixture]

    expected = [
        {"handler": "support", "request": "Need help with setup"},
        {"handler": "calculator", "request": "calculate batch gradient"},
        {"handler": "support", "request": "Help calculate cross-entropy"}, # Ambiguous: support > calculator
        {"handler": "fallback", "request": "play background music"}
    ]

    with open(os.path.join(work_dir, "results.json"), "w") as f:
        json.dump({"expected": expected, "actual": actual_results}, f, indent=2)

    for act, exp in zip(actual_results, expected):
        assert act["selected_handler"] == exp["handler"], f"Failed for {exp['request']}: got {act['selected_handler']} expected {exp['handler']}"

    assert len(router.audit_log) == 4, "Audit log must contain all 4 decisions"

    print("All checks passed successfully for R38-T005.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
