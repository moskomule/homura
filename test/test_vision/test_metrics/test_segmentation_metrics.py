from unittest import TestCase
import torch
from homura.vision.metrics import segmentation_metrics


class TestSegmentation_metrics(TestCase):
    def test_segmentation_metrics(self):
        output = torch.zeros(1, 2, 4, 4)
        target = torch.zeros(1, 4, 4)
        output[0, 0, 2:4, :] = 1
        output[0, 1, 0:2, :] = 1
        target[0, :, 0:3] = 1
        # output target
        # 1100   1111
        # 1100   1111
        # 1100   1111
        # 1100   0000
        metrics = segmentation_metrics(output, target.long())
        class_iou = metrics["class_iou"]
        self.assertAlmostEqual(class_iou[0], 1 / 5)
        self.assertAlmostEqual(class_iou[1], 3 / 7)
        self.assertAlmostEqual(metrics["mean_iou"].item(), 11 / 35)
