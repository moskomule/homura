from unittest import TestCase

import torch

from homura.metrics.confusion_matrix import ConfusionMatrix


class CMTest(TestCase):
    def test(self):
        cm = ConfusionMatrix(num_classes=3)
        pred = torch.tensor([0, 1, 1, 2])
        gt = torch.tensor([[2, 0, 0, 1]])
        expected = torch.tensor([[0, 0, 1],
                                 [2, 0, 0],
                                 [0, 1, 0]])
        cm.update(pred, gt)
        self.assertTrue((expected == cm.matrix).all())
