import unittest
import numpy as np
import torch

from homura.utils.reporter.wrapper import ReporterWrapper, TQDMWrapper


class TestReporter(unittest.TestCase):
    def test_tensor_type_check(self):
        r = ReporterWrapper(".")
        nptensor = np.zeros([3, 2])
        tensor = torch.zeros(3, 2)
        self.assertEqual(r._tensor_type_check(nptensor), (nptensor, 2))
        self.assertEqual(r._tensor_type_check(tensor), (tensor, 2))

    def test_run(self):
        with TQDMWrapper(range(1)) as tr:
            for i in tr:
                tr.add_text("test", "test_text", i)
                tr.add_scalar(i, "test_scalar", i)
                tr.add_scalars({"v1": i + 1,
                                "v2": i + 2}, "test_scalars", i)
