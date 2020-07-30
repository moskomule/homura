import hydra
import torch
import torch.nn.functional as F

from homura import optim, lr_scheduler, callbacks, reporters, trainers
from homura.vision import MODEL_REGISTRY, DATASET_REGISTRY


@hydra.main('config/cifar10.yaml')
def main(cfg):
    model = MODEL_REGISTRY(cfg.model.name)(num_classes=10)
    train_loader, test_loader = DATASET_REGISTRY("cifar10")(cfg.data.batch_size)
    optimizer = None if cfg.bn_no_wd else optim.SGD(lr=1e-1, momentum=0.9, weight_decay=cfg.optim.weight_decay)
    scheduler = lr_scheduler.MultiStepLR([100, 150], gamma=cfg.optim.lr_decay)
    tq = reporters.TQDMReporter(range(cfg.optim.epochs), verb=True)
    c = [callbacks.AccuracyCallback(),
         callbacks.LossCallback(),
         reporters.IOReporter("."),
         reporters.TensorboardReporter("."),
         callbacks.WeightSave("."),
         tq]

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
                                    callbacks=c,
                                    scheduler=scheduler,
                                    use_amp=cfg.use_amp) as trainer:

        for _ in tq:
            trainer.train(train_loader)
            trainer.test(test_loader)


if __name__ == '__main__':
    main()
