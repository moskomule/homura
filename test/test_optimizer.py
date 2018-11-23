from unittest import TestCase

from torch import nn

from homura.optim import *


class TestSchedulers(TestCase):
    def test_optimizer(self):
        model = nn.Linear(10, 10)
        optimizer = SGD(lr=0.1)
        optimizer.set_model(model.parameters())
        optimizer.optim
