"""
Unit tests for Applied Lab R38-T008: tensor-shape contract & failure recovery.
"""

import unittest
from example import AutoencoderBatchPreprocessor


class TestAutoencoderPreprocessing(unittest.TestCase):

    def setUp(self):
        self.processor = AutoencoderBatchPreprocessor()

    def test_canonical_fixture_shape_and_values(self):
        """
        Batch [[[0, 255], [128, 64]]] stays shape (1, 2, 2);
        0 -> 0.0 and 255 -> 1.0; valid intermediate values normalized.
        """
        batch = [[[0, 255], [128, 64]]]
        norm = self.processor.normalize(batch)
        shape = self.processor.get_shape(norm)
        
        self.assertEqual(shape, (1, 2, 2))
        self.assertAlmostEqual(norm[0][0][0], 0.0, places=6)
        self.assertAlmostEqual(norm[0][0][1], 1.0, places=6)
        self.assertAlmostEqual(norm[0][1][0], 128 / 255.0, places=6)
        self.assertAlmostEqual(norm[0][1][1], 64 / 255.0, places=6)

    def test_failure_and_subsequent_recovery(self):
        """
        Exercise invalid input, verify error logging, then demonstrate
        flawless recovery on the next valid batch.
        """
        invalid_batch = [[[0, -99], [128, 64]]]
        valid_batch = [[[0, 255], [128, 64]]]

        # Step 1: Process invalid
        res1 = self.processor.safe_process_with_recovery(invalid_batch)
        self.assertFalse(res1["success"])
        self.assertIn("out of valid range", res1["error"])
        self.assertEqual(len(self.processor.error_log), 1)

        # Step 2: Recover on valid
        res2 = self.processor.safe_process_with_recovery(valid_batch)
        self.assertTrue(res2["success"])
        self.assertEqual(res2["shape"], (1, 2, 2))
        self.assertAlmostEqual(res2["data"][0][0][1], 1.0)

    def test_missing_data_rejected(self):
        """Missing pixel (None) is rejected with clear error message."""
        batch_missing = [[[0, None], [128, 64]]]
        with self.assertRaises(ValueError):
            self.processor.normalize(batch_missing)

    def test_pixel_greater_than_255_rejected(self):
        """Pixel > 255 is rejected with ValueError."""
        batch = [[[0, 300], [128, 64]]]
        with self.assertRaises(ValueError):
            self.processor.normalize(batch)

    def test_deliberate_wrong_result_fails(self):
        """Deliberately wrong result triggers assertion failure."""
        batch = [[[0, 255], [128, 64]]]
        norm = self.processor.normalize(batch)
        with self.assertRaises(AssertionError):
            self.assertEqual(norm[0][0][1], 0.0)


if __name__ == "__main__":
    unittest.main()
