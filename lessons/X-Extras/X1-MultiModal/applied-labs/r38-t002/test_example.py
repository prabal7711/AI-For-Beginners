"""
Unit tests for Applied Lab R38-T002: classification evaluation & zero-denominator policy.
"""

import unittest
from example import evaluate_binary_classification


class TestClassificationEvaluation(unittest.TestCase):

    def test_canonical_fixture(self):
        """
        Canonical fixture:
        y_true=[1, 1, 0, 0], y_pred=[1, 0, 1, 0] gives
        TP=1, FN=1, FP=1, TN=1, precision=0.5, recall=0.5
        """
        y_true = [1, 1, 0, 0]
        y_pred = [1, 0, 1, 0]
        res = evaluate_binary_classification(y_true, y_pred)
        
        self.assertEqual(res["TP"], 1)
        self.assertEqual(res["FN"], 1)
        self.assertEqual(res["FP"], 1)
        self.assertEqual(res["TN"], 1)
        self.assertAlmostEqual(res["precision"], 0.5, places=6)
        self.assertAlmostEqual(res["recall"], 0.5, places=6)

    def test_zero_denominator_policy_no_positive_predictions(self):
        """
        When there are no positive predictions (TP=0, FP=0),
        precision must yield 0.0 under the documented policy.
        """
        y_true = [1, 1, 0, 0]
        y_pred = [0, 0, 0, 0]
        res = evaluate_binary_classification(y_true, y_pred)
        self.assertEqual(res["TP"], 0)
        self.assertEqual(res["FP"], 0)
        self.assertEqual(res["precision"], 0.0)
        self.assertEqual(res["recall"], 0.0)

    def test_zero_denominator_policy_no_actual_positives(self):
        """
        When there are no actual positive labels (TP=0, FN=0),
        recall must yield 0.0 under the documented policy.
        """
        y_true = [0, 0, 0, 0]
        y_pred = [1, 0, 0, 0]
        res = evaluate_binary_classification(y_true, y_pred)
        self.assertEqual(res["TP"], 0)
        self.assertEqual(res["FN"], 0)
        self.assertEqual(res["recall"], 0.0)

    def test_smallest_valid_boundary_case(self):
        """Single element input."""
        res_single_match = evaluate_binary_classification([1], [1])
        self.assertEqual(res_single_match["TP"], 1)
        self.assertEqual(res_single_match["precision"], 1.0)
        self.assertEqual(res_single_match["recall"], 1.0)

        res_single_neg = evaluate_binary_classification([0], [0])
        self.assertEqual(res_single_neg["TN"], 1)
        self.assertEqual(res_single_neg["precision"], 0.0)
        self.assertEqual(res_single_neg["recall"], 0.0)

    def test_perfect_classification(self):
        """All correct predictions."""
        y_true = [1, 1, 0, 0]
        y_pred = [1, 1, 0, 0]
        res = evaluate_binary_classification(y_true, y_pred)
        self.assertEqual(res["TP"], 2)
        self.assertEqual(res["TN"], 2)
        self.assertEqual(res["FP"], 0)
        self.assertEqual(res["FN"], 0)
        self.assertEqual(res["precision"], 1.0)
        self.assertEqual(res["recall"], 1.0)

    def test_invalid_cases(self):
        """Empty inputs, mismatched lengths, or non-binary elements raise errors."""
        with self.assertRaises(ValueError):
            evaluate_binary_classification([], [])
        with self.assertRaises(ValueError):
            evaluate_binary_classification([1, 0], [1])
        with self.assertRaises(ValueError):
            evaluate_binary_classification([1, 2], [1, 0])
        with self.assertRaises(TypeError):
            evaluate_binary_classification("not a list", [1, 0])

    def test_deliberate_wrong_result_fails(self):
        """Deliberately wrong result triggers assertion failure."""
        y_true = [1, 1, 0, 0]
        y_pred = [1, 0, 1, 0]
        res = evaluate_binary_classification(y_true, y_pred)
        with self.assertRaises(AssertionError):
            self.assertEqual(res["precision"], 1.0)


if __name__ == "__main__":
    unittest.main()
