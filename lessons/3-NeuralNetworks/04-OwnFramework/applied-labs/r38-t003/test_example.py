"""
Unit tests for Applied Lab R38-T003: single perceptron update & approach comparison.
"""

import unittest
from example import direct_perceptron_step, independent_perceptron_step, compare_approaches


class TestPerceptronUpdate(unittest.TestCase):

    def test_canonical_fixture_update(self):
        """
        Weights [0, 0], bias 0, input [1, 2], target 1, initial prediction 0:
        learning-rate 1 update gives weights [1.0, 2.0], bias 1.0.
        """
        w = [0.0, 0.0]
        b = 0.0
        x = [1.0, 2.0]
        target = 1
        lr = 1.0

        res = direct_perceptron_step(w, b, x, target, lr)
        self.assertEqual(res["initial_prediction"], 0)
        self.assertEqual(res["weights"], [1.0, 2.0])
        self.assertEqual(res["bias"], 1.0)

    def test_approaches_equivalence(self):
        """Direct and independent implementations must yield identical results."""
        w = [0.0, 0.0]
        b = 0.0
        x = [1.0, 2.0]
        target = 1
        lr = 1.0

        comp = compare_approaches(w, b, x, target, lr)
        self.assertTrue(comp["equivalent"])
        self.assertEqual(comp["direct"]["weights"], comp["reference"]["weights"])
        self.assertEqual(comp["direct"]["bias"], comp["reference"]["bias"])

    def test_zero_score_tie_rule(self):
        """Score exactly 0 must predict 0 under documented tie rule."""
        res_zero = direct_perceptron_step([0.0, 0.0], 0.0, [5.0, 10.0], target=0)
        self.assertEqual(res_zero["score"], 0.0)
        self.assertEqual(res_zero["initial_prediction"], 0)
        # Target 0 and prediction 0 means no error and no update
        self.assertEqual(res_zero["weights"], [0.0, 0.0])
        self.assertEqual(res_zero["bias"], 0.0)

    def test_positive_score_predicts_one(self):
        """Score > 0 predicts 1."""
        res = direct_perceptron_step([1.0, 1.0], 0.0, [1.0, 1.0], target=1)
        self.assertEqual(res["score"], 2.0)
        self.assertEqual(res["initial_prediction"], 1)
        self.assertEqual(res["error"], 0)

    def test_invalid_inputs(self):
        """Mismatched dimension or invalid target raises ValueError."""
        with self.assertRaises(ValueError):
            direct_perceptron_step([1.0], 0.0, [1.0, 2.0], target=1)
        with self.assertRaises(ValueError):
            direct_perceptron_step([1.0, 2.0], 0.0, [1.0, 2.0], target=3)

    def test_deliberate_wrong_result_fails(self):
        """Deliberately wrong result triggers assertion failure."""
        res = direct_perceptron_step([0.0, 0.0], 0.0, [1.0, 2.0], target=1, learning_rate=1.0)
        with self.assertRaises(AssertionError):
            self.assertEqual(res["weights"], [999.0, 999.0])


if __name__ == "__main__":
    unittest.main()
