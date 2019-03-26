import pytest
import torch
from homura import optim, is_apex_available, trainers, utils
from torch import nn
from torch.nn import functional as F


@pytest.mark.skipif(not (torch.cuda.is_available() and is_apex_available), reason="GPU and apex is unavailable")
def test_fp16():
    model = nn.Linear(10, 10)
    optimizer = optim.SGD(lr=0.1)
    trainer = trainers.AMPTrainer(model, optimizer, F.cross_entropy)
    epoch = range(1)
    loader = [(torch.randn(2, 10), torch.zeros(2, dtype=torch.long)) for _ in range(10)]
    for _ in epoch:
        trainer.train(loader)
        trainer.test(loader)


def test_dict_model():
    # test if model and optimizer are dict
    class Trainer(trainers.TrainerBase):
        def iteration(self, data):
            input, target = data
            output = self.model["generator"](input) + self.model["discriminator"](input)
            loss = self.loss_f(output, target)
            results = utils.Map(loss=loss, output=output)
            if self.is_train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            return results

    model = {"generator": nn.Linear(10, 10),
             "discriminator": nn.Linear(10, 10)}
    optimizer = {"generator": optim.SGD(lr=0.1),
                 "discriminator": None}
    trainer = Trainer(model, optimizer, F.cross_entropy)
    loader = [(torch.randn(2, 10), torch.zeros(2, dtype=torch.long)) for _ in range(10)]
    for _ in range(1):
        trainer.train(loader)
        trainer.test(loader)
