from __future__ import annotations

import unittest

from evaluation.confusion_cases import Box, box_iou, match_boxes


class ConfusionCaseTests(unittest.TestCase):
    def test_box_iou(self) -> None:
        self.assertAlmostEqual(box_iou((0, 0, 10, 10), (5, 5, 15, 15)), 25 / 175)

    def test_matching_counts_true_positive_and_false_negative(self) -> None:
        truth = [Box(0, "person", (0, 0, 10, 10))]
        prediction = [Box(0, "person", (1, 1, 11, 11), 0.9)]
        counts, cases = match_boxes(truth, prediction)
        self.assertEqual(counts["tp:person"], 1)
        self.assertFalse(cases)

    def test_class_confusion_is_explained(self) -> None:
        truth = [Box(1, "carton", (0, 0, 10, 10))]
        prediction = [Box(0, "person", (0, 0, 10, 10), 0.8)]
        counts, cases = match_boxes(truth, prediction)
        self.assertEqual(counts["fp:person"], 1)
        self.assertEqual(counts["fn:carton"], 1)
        self.assertEqual(cases[0]["reason"], "class_confusion")


if __name__ == "__main__":
    unittest.main()
