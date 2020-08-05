import hydra
import torch
import torch.nn.functional as F

from homura import optim, lr_scheduler, reporters, trainers
from homura.vision import MODEL_REGISTRY, DATASET_REGISTRY


@hydra.main('config/cifar10.yaml')
def main(cfg):
    model = MODEL_REGISTRY(cfg.model.name)(num_classes=10)
    train_loader, test_loader = DATASET_REGISTRY("cifar10")(cfg.data.batch_size, num_workers=4)
    optimizer = None if cfg.bn_no_wd else optim.SGD(lr=1e-1, momentum=0.9, weight_decay=cfg.optim.weight_decay)
    scheduler = lr_scheduler.MultiStepLR([100, 150], gamma=cfg.optim.lr_decay)

    if cfg.bn_no_wd:
        def set_optimizer(trainer):
            bn_params = []
            non_bn_parameters = []
            for name, p in trainer.model.named_parameters():
                if "bn" in name:
                    bn_params.append(p)
                else:
                    non_bn_parameters.append(p)
            optim_params = [
                {"params": bn_params, "weight_decay": 0},
                {"params": non_bn_parameters, "weight_decay": cfg.optim.weight_decay},
            ]
            trainer.optimizer = torch.optim.SGD(optim_params, lr=1e-1, momentum=0.9)

        trainers.SupervisedTrainer.set_optimizer = set_optimizer

    with trainers.SupervisedTrainer(model,
                                    optimizer,
                                    F.cross_entropy,
                                    reporters=[reporters.TensorboardReporter('.')],
                                    scheduler=scheduler,
                                    use_amp=cfg.use_amp) as trainer:

        for _ in trainer.epoch_range(cfg.optim.epochs):
            trainer.train(train_loader)
            trainer.test(test_loader)


if __name__ == '__main__':
    main()
