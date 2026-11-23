"""
Unit tests for Applied Lab R38-T009: single perceptron update & configuration evaluation.
"""

import unittest
from example import perceptron_update_step, run_configuration_experiment


class TestPerceptronExperiment(unittest.TestCase):

    def test_canonical_fixture_step(self):
        """
        Weights [0, 0], bias 0, input [1, 2], target 1, initial prediction 0:
        learning-rate 1 update gives weights [1.0, 2.0], bias 1.0.
        """
        w = [0.0, 0.0]
        b = 0.0
        x = [1.0, 2.0]
        target = 1
        lr = 1.0

        res = perceptron_update_step(w, b, x, target, lr)
        self.assertEqual(res["prediction"], 0)
        self.assertEqual(res["weights"], [1.0, 2.0])
        self.assertEqual(res["bias"], 1.0)
        self.assertAlmostEqual(res["post_score"], 6.0, places=6)

    def test_three_configurations_experiment(self):
        """Evaluate 3 controlled configurations (lr = 0.1, 0.5, 1.0)."""
        w = [0.0, 0.0]
        b = 0.0
        x = [1.0, 2.0]
        target = 1

        exp = run_configuration_experiment(w, b, x, target, [0.1, 0.5, 1.0])
        cfgs = exp["configurations"]

        self.assertIn("lr_0.1", cfgs)
        self.assertIn("lr_0.5", cfgs)
        self.assertIn("lr_1.0", cfgs)

        # Monotonic increase in step size
        self.assertLess(cfgs["lr_0.1"]["step_size"], cfgs["lr_0.5"]["step_size"])
        self.assertLess(cfgs["lr_0.5"]["step_size"], cfgs["lr_1.0"]["step_size"])

        # Correctness of weights
        self.assertEqual(cfgs["lr_0.1"]["updated_weights"], [0.1, 0.2])
        self.assertEqual(cfgs["lr_0.5"]["updated_weights"], [0.5, 1.0])
        self.assertEqual(cfgs["lr_1.0"]["updated_weights"], [1.0, 2.0])

    def test_zero_score_tie_rule(self):
        """Score <= 0 must predict 0."""
        res = perceptron_update_step([0.0, 0.0], 0.0, [1.0, 1.0], target=0)
        self.assertEqual(res["score"], 0.0)
        self.assertEqual(res["prediction"], 0)
        self.assertEqual(res["error"], 0)
        self.assertEqual(res["weights"], [0.0, 0.0])

    def test_invalid_arguments(self):
        """Dimension mismatch or invalid target raises ValueError."""
        with self.assertRaises(ValueError):
            perceptron_update_step([1.0], 0.0, [1.0, 2.0], target=1)
        with self.assertRaises(ValueError):
            perceptron_update_step([1.0, 2.0], 0.0, [1.0, 2.0], target=5)

    def test_deliberate_wrong_result_fails(self):
        """Deliberately wrong result triggers assertion failure."""
        res = perceptron_update_step([0.0, 0.0], 0.0, [1.0, 2.0], target=1, learning_rate=1.0)
        with self.assertRaises(AssertionError):
            self.assertEqual(res["weights"], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
