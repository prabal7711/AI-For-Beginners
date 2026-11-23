"""
Unit tests for Applied Lab R38-T010: bounding-box intersection & scenario assessment.
"""

import unittest
from example import compute_iou, validate_box, scenario_assessment


class TestBoundingBoxScenarioAssessment(unittest.TestCase):

    def test_canonical_fixture(self):
        """Boxes [0, 0, 2, 2] and [1, 1, 3, 3] have IoU = 1/7."""
        iou = compute_iou([0, 0, 2, 2], [1, 1, 3, 3])
        self.assertAlmostEqual(iou, 1.0 / 7.0, places=6)

    def test_identical_and_disjoint_boxes(self):
        """Identical boxes have 1.0; disjoint boxes have 0.0."""
        self.assertAlmostEqual(compute_iou([1, 1, 4, 4], [1, 1, 4, 4]), 1.0, places=6)
        self.assertAlmostEqual(compute_iou([0, 0, 1, 1], [5, 5, 6, 6]), 0.0, places=6)

    def test_applied_question_1_prediction(self):
        """Predict output check for [0, 0, 4, 4] and [2, 2, 6, 6]."""
        assessment = scenario_assessment()
        self.assertAlmostEqual(assessment["q1"]["answer"], 1.0 / 7.0, places=6)
        self.assertEqual(assessment["q1"]["expected"], 1.0 / 7.0)

    def test_applied_question_2_diagnosis(self):
        """Diagnose failure check for inverted box [4, 5, 2, 8]."""
        assessment = scenario_assessment()
        self.assertIn("Inverted", assessment["q2"]["diagnosed_error"])
        with self.assertRaises(ValueError):
            validate_box([4, 5, 2, 8])

    def test_applied_question_3_boundary_policy(self):
        """Boundary policy check for edge-touching adjacent boxes."""
        assessment = scenario_assessment()
        self.assertEqual(assessment["q3"]["answer"], 0.0)
        self.assertEqual(assessment["q3"]["expected"], 0.0)

    def test_deliberate_wrong_result_fails(self):
        """Deliberately wrong result triggers assertion failure."""
        iou = compute_iou([0, 0, 2, 2], [1, 1, 3, 3])
        with self.assertRaises(AssertionError):
            self.assertEqual(iou, 0.5)


if __name__ == "__main__":
    unittest.main()
