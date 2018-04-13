import unittest
import torch
from homura.utils import callbacks

output = torch.FloatTensor([[0.9, 0.1, 0, 0],
                            [0.8, 0.1, 0.1, 0],
                            [0.2, 0.1, 0.7, 0],
                            [0.1, 0.9, 0, 0]])
target = torch.LongTensor([0, 1, 2, 3])


class TestCallbacks(unittest.TestCase):
    def test_accuracy_callback(self):
        c = callbacks.AccuracyCallback(k=(1, 3))
        top1_3 = c.end_epoch({"output": output, "target": target})
        self.assertEqual(top1_3, [2 / 4, 4 / 4])

    def test_callback_list(self):
        c = callbacks.CallbackList(callbacks.AccuracyCallback())
        acc, *_ = c.end_epoch({"output": output, "target": target})
        self.assertEqual(acc, [2 / 4])
