import unittest
import numpy as np
import torch
from torch.autograd import Variable

from homura.utils.reporter import Reporter, TQDMReporter, TensorBoardReporter, VisdomReporter


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
