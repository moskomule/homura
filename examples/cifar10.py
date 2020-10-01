import hydra
import torch
import torch.nn.functional as F

from homura import enable_accimage, lr_scheduler, optim, reporters, trainers
from homura.vision import DATASET_REGISTRY, MODEL_REGISTRY


@hydra.main('config/cifar10.yaml')
def main(cfg):
    if cfg.use_accimage:
        enable_accimage()
    model = MODEL_REGISTRY(cfg.model.name)(num_classes=10)
    train_loader, test_loader = DATASET_REGISTRY("fast_cifar10" if cfg.use_fast_collate else "cifar10"
                                                 )(cfg.data.batch_size, num_workers=4,
                                                   use_prefetcher=cfg.use_prefetcher)
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

    if cfg.use_zerograd_none:
        import types

        def set_optimizer(trainer):
            # see Apex for details
            def zero_grad(self):
                for group in self.param_groups:
                    for p in group['params']:
                        p.grad = None

            trainer.optimizer = trainer.optimizer(trainer.model.parameters())
            trainer.optimizer.zero_grad = types.MethodType(zero_grad, trainer.optimizer)

        trainers.SupervisedTrainer.set_optimizer = set_optimizer

    with trainers.SupervisedTrainer(model,
                                    optimizer,
                                    F.cross_entropy,
                                    reporters=[reporters.TensorboardReporter('.')],
                                    scheduler=scheduler,
                                    use_amp=cfg.use_amp,
                                    debug=cfg.debug
                                    ) as trainer:

        for _ in trainer.epoch_range(cfg.optim.epochs):
            trainer.train(train_loader)
            trainer.test(test_loader)

        print(f"Max Test Accuracy={max(trainer.reporter.history('accuracy/test')):.3f}")


if __name__ == '__main__':
    main()
