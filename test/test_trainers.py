import pytest
import torch
from homura import optim, is_apex_available, trainers
from torch import nn
from torch.nn import functional as F


@pytest.mark.skipif(not (torch.cuda.is_available() and is_apex_available))
def test():
    model = nn.Linear(10, 10)
    optimizer = optim.SGD(lr=0.1)
    trainer = trainers.FP16Trainer(model, optimizer, F.cross_entropy)
    epoch = range(1)
    loader = [(torch.randn(2, 10), torch.zeros(2, dtype=torch.long)) for _ in range(10)]
    for _ in epoch:
        trainer.train(loader)
        trainer.test(loader)
