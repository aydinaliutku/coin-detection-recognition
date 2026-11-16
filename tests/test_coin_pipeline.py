import unittest
from pathlib import Path

import numpy as np

from src.coin_pipeline import classify, hog_descriptor, hough_circle_transform, label_from_filename


class CoinPipelineTests(unittest.TestCase):
    def test_hough_finds_a_known_center_at_a_single_radius(self):
        edges = np.zeros((70, 70), dtype=np.uint8)
        angles = np.deg2rad(np.arange(0, 360, 10))
        ring = np.rint(18 * np.column_stack([np.cos(angles), np.sin(angles)])).astype(int)
        edges[34 + ring[:, 1], 31 + ring[:, 0]] = 1
        circles = hough_circle_transform(edges, min_radius=18, max_radius=18, threshold=15)
        x, y, radius, _ = circles[0]
        self.assertLessEqual(abs(x - 31) + abs(y - 34), 2)
        self.assertEqual(radius, 18)

    def test_hog_has_expected_size_and_zero_flat_gradient(self):
        descriptor = hog_descriptor(np.full((64, 64), 127, dtype=np.uint8))
        self.assertEqual(descriptor.shape, (8100,))
        self.assertTrue(np.all(descriptor == 0))

    def test_train_test_directory_cannot_alias(self):
        with self.assertRaisesRegex(ValueError, "separate"):
            classify(Path("."), Path.cwd())

    def test_label_prefix_must_have_a_boundary(self):
        self.assertIsNotNone(label_from_filename(Path("1TL_reverse_001.png")))
        self.assertIsNone(label_from_filename(Path("1TL_reverseCoin.png")))
