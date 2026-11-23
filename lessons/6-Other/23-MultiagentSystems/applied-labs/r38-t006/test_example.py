"""
Unit tests for Applied Lab R38-T006: dependency-plan validation & refactoring equivalence.
"""

import unittest
from example import (
    baseline_validate_and_order,
    refactored_validate_and_order,
    MissingDependencyError,
    CyclicDependencyError,
)


class TestDependencyPlanValidation(unittest.TestCase):

    def setUp(self):
        self.canonical_plan = {
            "A": [],
            "B": ["A"],
            "C": ["B"]
        }

    def test_canonical_ordering(self):
        """A has no dependencies, B depends on A, C on B: order is ['A', 'B', 'C']."""
        order = refactored_validate_and_order(self.canonical_plan)
        self.assertEqual(order, ["A", "B", "C"])

    def test_before_after_refactoring_equivalence(self):
        """Baseline and refactored versions produce identical results."""
        b_res = baseline_validate_and_order(self.canonical_plan)
        r_res = refactored_validate_and_order(self.canonical_plan)
        self.assertEqual(b_res, r_res)

    def test_cyclic_dependency_rejected(self):
        """A -> B -> A is rejected with CyclicDependencyError."""
        cyclic_plan = {
            "A": ["B"],
            "B": ["A"]
        }
        with self.assertRaises(CyclicDependencyError):
            refactored_validate_and_order(cyclic_plan)
        with self.assertRaises(CyclicDependencyError):
            baseline_validate_and_order(cyclic_plan)

    def test_unknown_dependency_reported_by_id(self):
        """Unknown dependency is reported by task ID in exception message."""
        invalid_plan = {
            "A": [],
            "B": ["UNKNOWN_TASK_99"]
        }
        with self.assertRaises(MissingDependencyError) as ctx:
            refactored_validate_and_order(invalid_plan)
        self.assertIn("UNKNOWN_TASK_99", str(ctx.exception))

    def test_independent_tasks(self):
        """Independent tasks are ordered deterministically."""
        plan = {"Y": [], "X": []}
        order = refactored_validate_and_order(plan)
        self.assertEqual(order, ["X", "Y"])

    def test_deliberate_wrong_result_fails(self):
        """Deliberately wrong result triggers assertion failure."""
        order = refactored_validate_and_order(self.canonical_plan)
        with self.assertRaises(AssertionError):
            self.assertEqual(order, ["C", "B", "A"])


if __name__ == "__main__":
    unittest.main()
