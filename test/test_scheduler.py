from unittest import TestCase

from torch import nn, optim

from homura.lr_scheduler import *


class TestSchedulers(TestCase):
    def test_scheduler(self):
        model = nn.Linear(10, 10)
        opt = optim.SGD(model.parameters(), lr=0.1)
        scheduler = StepLR(step_size=10)
        scheduler.set_optimizer(opt)
