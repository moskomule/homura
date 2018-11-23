from unittest import TestCase

from torch import nn
from homura import trainer, optim


class TestSchedulers(TestCase):
    def test_supervised_trainer_init(self):
        model = nn.Linear(10, 10)
        opt = optim.SGD(lr=0.1)
        _trainer = trainer.SupervisedTrainer(model, opt, nn.CrossEntropyLoss())