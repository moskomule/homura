import torch
from torch import nn
from torch.nn import functional as F

from homura import optim, trainers, utils, lr_scheduler


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


def test_update_scheduler():
    model = nn.Linear(10, 10)
    optimizer = optim.SGD(lr=0.1)
    trainer = trainers.SupervisedTrainer(model, optimizer, F.cross_entropy)
    trainer.update_scheduler(lr_scheduler.LambdaLR(lambda step: 0.1 ** step),
                             update_scheduler_by_epoch=False)
    loader = [(torch.randn(2, 10), torch.zeros(2, dtype=torch.long)) for _ in range(2)]
    trainer.train(loader)
    # 0.1 * (0.1 ** 2)
    assert list(trainer.optimizer.param_groups)[0]['lr'] == 0.1 ** 3

    trainer.update_scheduler(lr_scheduler.LambdaLR(lambda epoch: 0.1 ** epoch, last_epoch=1),
                             update_scheduler_by_epoch=True)
    trainer.train(loader)
    assert list(trainer.optimizer.param_groups)[0]['lr'] == 0.1 ** 3
