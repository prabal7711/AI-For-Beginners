"""
Unit tests for Applied Lab R38-T004: bounding-box intersection & state trace.
"""

import unittest
from example import compute_iou_with_trace, validate_box


class TestBoundingBoxIoU(unittest.TestCase):

    def test_canonical_fixture_overlap(self):
        """
        Boxes [0, 0, 2, 2] and [1, 1, 3, 3] have IoU = 1/7 (~0.142857).
        """
        res = compute_iou_with_trace([0, 0, 2, 2], [1, 1, 3, 3])
        self.assertAlmostEqual(res["iou"], 1.0 / 7.0, places=6)
        self.assertEqual(res["area_a"], 4.0)
        self.assertEqual(res["area_b"], 4.0)
        self.assertEqual(res["intersection_area"], 1.0)
        self.assertEqual(res["union_area"], 7.0)
        self.assertTrue(res["invariant_holds"])

    def test_identical_boxes(self):
        """Identical boxes have IoU = 1.0."""
        res = compute_iou_with_trace([0, 0, 5, 5], [0, 0, 5, 5])
        self.assertAlmostEqual(res["iou"], 1.0, places=6)
        self.assertTrue(res["invariant_holds"])

    def test_disjoint_boxes(self):
        """Disjoint boxes have IoU = 0.0."""
        res = compute_iou_with_trace([0, 0, 2, 2], [10, 10, 12, 12])
        self.assertAlmostEqual(res["iou"], 0.0, places=6)
        self.assertEqual(res["intersection_area"], 0.0)
        self.assertEqual(res["union_area"], 8.0)
        self.assertTrue(res["invariant_holds"])

    def test_inverted_boxes_rejected(self):
        """Inverted boxes (x1 > x2 or y1 > y2) are rejected with ValueError."""
        with self.assertRaises(ValueError):
            compute_iou_with_trace([3, 0, 1, 2], [0, 0, 2, 2])
        with self.assertRaises(ValueError):
            compute_iou_with_trace([0, 4, 2, 1], [0, 0, 2, 2])

    def test_invalid_input_types(self):
        """Invalid types or lengths raise appropriate errors."""
        with self.assertRaises(TypeError):
            validate_box("not a box")
        with self.assertRaises(ValueError):
            validate_box([0, 0, 1])

    def test_deliberate_wrong_result_fails(self):
        """Deliberately wrong result triggers assertion failure."""
        res = compute_iou_with_trace([0, 0, 2, 2], [1, 1, 3, 3])
        with self.assertRaises(AssertionError):
            self.assertEqual(res["iou"], 0.5)


if __name__ == "__main__":
    unittest.main()
