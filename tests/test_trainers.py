import pytest
import torch
from torch import nn
from torch.nn import functional as F

from homura import trainers, optim, lr_scheduler


def test_dict_model():
    # test if model and optimizer are dict
    class Trainer(trainers.TrainerBase):
        def iteration(self, data):
            input, target = data
            output = self.model["generator"](input) + self.model["discriminator"](input)
            loss = self.loss_f(output, target)
            if self.is_train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.reporter('loss', loss.detach_())

    model = {"generator": nn.Linear(10, 10),
             "discriminator": nn.Linear(10, 10)}
    optimizer = {"generator": torch.optim.SGD(model["generator"].parameters(), lr=0.1),
                 "discriminator": None}
    trainer = Trainer(model, optimizer, F.cross_entropy)
    loader = [(torch.randn(2, 10), torch.zeros(2, dtype=torch.long)) for _ in range(10)]

    for _ in trainer.epoch_range(1):
        trainer.train(loader)
        trainer.test(loader)


def test_basic_trainer():
    model = nn.Linear(10, 10)
    optimizer = optim.SGD()
    scheduler = lr_scheduler.StepLR(9)
    trainer = trainers.SupervisedTrainer(model, optimizer, F.cross_entropy, scheduler=scheduler,
                                         update_scheduler_by_epoch=False)
    loader = [(torch.randn(2, 10), torch.zeros(2, dtype=torch.long)) for _ in range(10)]
    for _ in trainer.epoch_range(1):
        trainer.train(loader)
    assert pytest.approx(trainer.optimizer.param_groups[0]["lr"], 0.01)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
    trainer = trainers.SupervisedTrainer(model, optimizer, F.cross_entropy, scheduler=scheduler,
                                         update_scheduler_by_epoch=False)
    for _ in trainer.epoch_range(1):
        trainer.train(loader)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 9)
    trainer = trainers.SupervisedTrainer(model, optimizer, F.cross_entropy, scheduler=scheduler,
                                         update_scheduler_by_epoch=False)
    trainer.run(loader, loader, 15, 11)
    assert trainer.step == 11 - 1


def test_trainer_grad_accumulate():
    model = nn.Linear(10, 10)
    loader = [(torch.randn(4, 10), torch.zeros(4, dtype=torch.long)) for _ in range(10)]

    class _Optimizer(object):
        def step(self, *args, **kwargs): ...

        def zero_grad(self): ...

    class _Trainer(trainers.SupervisedTrainer):
        def set_optimizer(self
                          ) -> None:
            self.optimizer = _Optimizer()

    trainer = _Trainer(model, None, F.cross_entropy, quiet=True)
    for _ in trainer.epoch_range(1):
        trainer.train(loader)
    grads0 = [p.grad.data for p in model.parameters()]
    model.zero_grad()

    trainer = _Trainer(model, None, F.cross_entropy, grad_accum_steps=2, quiet=True)
    for _ in trainer.epoch_range(1):
        trainer.train(loader)
    grads1 = [p.grad.data for p in model.parameters()]

    assert all([torch.allclose(p0, p1) for p0, p1 in zip(grads0, grads1)])
