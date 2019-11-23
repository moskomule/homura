import torch
from torch import nn

from homura.callbacks import AccuracyCallback, TensorboardReporter
from homura.optim import SGD
from homura.trainers import SupervisedTrainer


def test_reporters():
    loader = [(torch.randn(2, 10), torch.zeros(2, dtype=torch.long)) for _ in range(10)]
    model = nn.Linear(10, 10)
    with SupervisedTrainer(model, SGD(lr=0.1), nn.CrossEntropyLoss(),
                           callbacks=[AccuracyCallback(), TensorboardReporter(".")]) as t:
        t.train(loader)
        t.test(loader)
