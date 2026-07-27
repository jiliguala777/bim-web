import unittest

import torch

from training.losses import CombinedSegmentationLoss, dice_per_class


class TrainingLossTests(unittest.TestCase):
    def test_perfect_prediction_reports_present_classes_and_absent_classes(self):
        target = torch.tensor([[[0, 1], [2, 3]]], dtype=torch.long)
        logits = torch.full((1, 4, 2, 2), -12.0)
        logits.scatter_(1, target.unsqueeze(1), 12.0)

        report = dice_per_class(logits, target)

        for class_id in range(4):
            self.assertTrue(report[class_id]["present"])
            self.assertGreater(report[class_id]["dice"], 0.999)

        background = torch.zeros((1, 2, 2), dtype=torch.long)
        absent_report = dice_per_class(
            torch.zeros((1, 4, 2, 2), dtype=torch.float32),
            background,
        )
        self.assertFalse(absent_report[1]["present"])
        self.assertIsNone(absent_report[1]["dice"])

    def test_combined_loss_is_finite_and_backpropagates_on_background_only(self):
        logits = torch.randn((1, 4, 8, 8), requires_grad=True)
        target = torch.zeros((1, 8, 8), dtype=torch.long)
        criterion = CombinedSegmentationLoss(class_weights=[0.2, 1.0, 1.5, 1.5])

        loss = criterion(logits, target)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_class_weights_from_report_are_bounded_and_serializable(self):
        weights = CombinedSegmentationLoss.weights_from_class_pixels(
            {"0": 250000, "1": 10000, "2": 500, "3": 0}
        )

        self.assertEqual(len(weights), 4)
        self.assertTrue(all(0.1 <= value <= 10.0 for value in weights))
        self.assertGreater(weights[2], weights[1])
        self.assertEqual(weights[3], 1.0)


if __name__ == "__main__":
    unittest.main()
