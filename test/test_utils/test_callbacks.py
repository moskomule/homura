import unittest
import torch
from homura.utils import callbacks

output = torch.FloatTensor([[0.9, 0.1, 0, 0],
                            [0.8, 0.1, 0.1, 0],
                            [0.2, 0.1, 0.7, 0],
                            [0.1, 0.9, 0, 0]])
target = torch.LongTensor([0, 1, 2, 3])


class TestCallbacks(unittest.TestCase):

    def test_callback_list(self):
        c = callbacks.CallbackList(callbacks.AccuracyCallback(1))
        acc, *_ = c.end_iteration({"output": output, "target": target, "name": "train"})
        self.assertEqual(acc["accuracy_train"], 2 / 4)

    def test_accuracy_callback(self):
        c = callbacks.AccuracyCallback(k=(1, 3))
        iter_acc = c.end_iteration({"output": output, "target": target, "name": "train"})
        self.assertEqual(iter_acc["accuracy_train_top1"], 2 / 4)
        self.assertEqual(iter_acc["accuracy_train_top3"], 4 / 4)

        c.end_iteration({"output": output[:2], "target": target[:2], "name": "train"})
        epoch_acc = c.end_epoch({"name": "train", "iter_per_epoch": 2})
        self.assertEqual(epoch_acc["accuracy_train_top1"], 3 / 6)

    def test_loss_callback(self):
        c = callbacks.LossCallback()
        iter_loss = c.end_iteration({"loss": 0.1, "name": "test"})
        self.assertEqual(iter_loss["loss_test"], 0.1)
