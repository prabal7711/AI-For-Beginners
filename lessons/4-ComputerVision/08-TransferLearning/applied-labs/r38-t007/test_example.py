"""
Unit tests for Applied Lab R38-T007: training-only imputation & leakage prevention.
"""

import unittest
from example import MeanImputer


class TestTrainingOnlyImputation(unittest.TestCase):

    def test_canonical_fixture_imputation(self):
        """
        Training [2, None, 4] yields mean 3.0;
        validation [100, None] becomes [100.0, 3.0], NOT [100.0, 100.0].
        """
        train = [2, None, 4]
        val = [100, None]

        imputer = MeanImputer()
        imputer.fit(train)
        self.assertEqual(imputer.fitted_mean, 3.0)

        res_val = imputer.transform(val)
        self.assertEqual(res_val, [100.0, 3.0])
        # Confirm it did NOT leak validation distribution
        self.assertNotEqual(res_val, [100.0, 100.0])

    def test_all_missing_training_input_rejected(self):
        """All-missing training input is rejected with ValueError."""
        imputer = MeanImputer()
        with self.assertRaises(ValueError):
            imputer.fit([None, None, None])

    def test_empty_training_input_rejected(self):
        """Empty list raises ValueError."""
        imputer = MeanImputer()
        with self.assertRaises(ValueError):
            imputer.fit([])

    def test_transform_before_fit_raises(self):
        """Calling transform() prior to fit() raises RuntimeError."""
        imputer = MeanImputer()
        with self.assertRaises(RuntimeError):
            imputer.transform([1, None])

    def test_float_precision(self):
        """Handles float observations correctly."""
        train = [1.5, None, 2.5]
        imputer = MeanImputer().fit(train)
        self.assertEqual(imputer.fitted_mean, 2.0)
        self.assertEqual(imputer.transform([None]), [2.0])

    def test_deliberate_wrong_result_fails(self):
        """Deliberately wrong result triggers assertion failure."""
        imputer = MeanImputer().fit([2, None, 4])
        res = imputer.transform([100, None])
        with self.assertRaises(AssertionError):
            self.assertEqual(res, [100.0, 999.0])


if __name__ == "__main__":
    unittest.main()
