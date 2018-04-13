import unittest
import numpy as np
import torch
from torch.autograd import Variable

from homura.utils.reporter import Reporter, TQDMReporter, TensorBoardReporter, VisdomReporter, ListReporter


class TestReporter(unittest.TestCase):
    def test_tensor_type_check(self):
        r = Reporter(".")
        nptensor = np.zeros([3, 2])
        tensor = torch.zeros(3, 2)
        variable = Variable(tensor)
        self.assertEqual(r._tensor_type_check(nptensor), (nptensor, 2))
        self.assertEqual(r._tensor_type_check(tensor), (tensor, 2))
        self.assertEqual(r._tensor_type_check(variable), (tensor, 2))

        with self.assertRaises(TypeError):
            r._tensor_type_check([[1, 2, 3], [1, 2, 3]])

    def test_run(self):
        with TQDMReporter(range(1)) as tr:
            for i in tr:
                tr.add_text("test", "test_text", i)
                tr.add_scalar(i, "test_scalar", i)
                tr.add_scalars({"v1": i + 1,
                                "v2": i + 2}, "test_scalars", i)

        with TensorBoardReporter() as tr:
            for i in range(1):
                tr.add_text("test", "test_text", i)
                tr.add_scalar(np.array([i]), "test_scalar", i)
                tr.add_scalars({"v1": np.array([i + 1]),
                                "v2": np.array([i + 2])}, "test_scalars", i)
                tr.add_image(torch.randn(3, 20, 20), "test_image", i)

        with VisdomReporter() as vr:
            for i in range(1):
                vr.add_text("test", "test_text", i)
                vr.add_scalar(torch.LongTensor([i]), "test_scalar", i)
                vr.add_scalars({"v1": torch.LongTensor([i]),
                                "v2": torch.LongTensor([i + 1])}, "test_scalars", i)
                vr.add_image((torch.randn(3, 20, 20)), "test_image", i)
                vr.add_images(torch.randn(4, 3, 20, 20), "test_images", i)

        with VisdomReporter() as vr, TQDMReporter(range(1)) as tr:
            lr = ListReporter(vr, tr)
            for i in tr:
                lr.add_text("test", "test_text", i)
