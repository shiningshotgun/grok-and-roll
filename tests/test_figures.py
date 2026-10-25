import os
import unittest
import xml.etree.ElementTree as ET


class TestFigures(unittest.TestCase):
    def _check(self, path):
        if not os.path.exists(path):
            self.skipTest(f"missing {path}")
        tree = ET.parse(path)
        root = tree.getroot()
        self.assertIn("viewBox", root.attrib)
        # plausible element count
        n = sum(1 for _ in root.iter())
        self.assertGreater(n, 10)

    def test_train_test_curves(self):
        self._check("figures/train_test_curves.svg")

    def test_progress_measures(self):
        self._check("figures/progress_measures.svg")

    def test_frequency_ablation(self):
        self._check("figures/frequency_ablation.svg")


if __name__ == "__main__":
    unittest.main()
