import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class CoinWorkflowTests(unittest.TestCase):
    def run_cli(self, *arguments):
        result = subprocess.run(
            [sys.executable, "src/coin_pipeline.py", *arguments],
            cwd=ROOT, capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_detection_and_classification_commands(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "detected.jpg"
            report = self.run_cli(
                "detect", "--image", "examples/coins.png", "--output", str(output),
                "--min-radius", "14", "--max-radius", "34", "--radius-step", "1",
                "--threshold", "17", "--sigma", "2.5",
            )
            self.assertRegex(report, r"Detected [1-9][0-9]* circles")
            self.assertIsNotNone(cv2.imread(str(output)))

            rng = np.random.default_rng(42)
            for split, count in (("train", 4), ("test", 2)):
                folder = root / split
                folder.mkdir()
                for label, endpoints in (
                    ("1TL_obverse", ((20, 20), (100, 100))),
                    ("50kr_reverse", ((20, 100), (100, 20))),
                ):
                    for index in range(count):
                        image = rng.integers(0, 30, (128, 128), dtype=np.uint8)
                        cv2.line(image, *endpoints, 230, 7)
                        self.assertTrue(cv2.imwrite(str(folder / f"{label}_{index}.png"), image))
            report = self.run_cli("classify", "--train", str(root / "train"), "--test", str(root / "test"))
            self.assertIn("Training images: 8 | test images: 4", report)
            self.assertIn("macro avg", report)
