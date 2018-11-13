from unittest import TestCase

import torch
from homura.modules import CategoricalConditionalBatchNorm


class TestCB(TestCase):
    def test_sanity(self):
        cb = CategoricalConditionalBatchNorm(10, 3)
        input = torch.randn(3, 10, 28, 28)
        cat = torch.LongTensor([2, 1, 0])
        out = cb(input, cat)
        self.assertEqual(out.shape, input.shape)
