import chika
import torch
import torch.nn.functional as F

from homura import enable_accimage, lr_scheduler, optim, reporters, trainers
from homura.vision import DATASET_REGISTRY, MODEL_REGISTRY


@chika.config
class Config:
    model: str = chika.choices("wrn28_2", "wrn28_10")
    batch_size: int = 128

    epochs: int = 200
    lr: float = 0.1
    weight_decay: float = 1e-4

    data: str = chika.choices("cifar10", "cifar100", "svhn")

    bn_no_wd: bool = False
    use_amp: bool = False
    use_accimage: bool = False
    use_multi_tensor: bool = False
    use_channel_last: bool = False
    prefetch_factor: int = 2
    persistent_workers: bool = False
    debug: bool = False
    download: bool = False


@chika.main(cfg_cls=Config)
def main(cfg):
    if cfg.use_accimage:
        enable_accimage()
    data = DATASET_REGISTRY(cfg.data).setup(cfg.batch_size, num_workers=4, download=cfg.download,
                                            prefetch_factor=cfg.prefetch_factor,
                                            persistent_workers=cfg.persistent_workers)
    model = MODEL_REGISTRY(cfg.model)(num_classes=data.num_classes)
    optimizer = None if cfg.bn_no_wd else optim.SGD(lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay,
                                                    multi_tensor=cfg.use_multi_tensor)
    scheduler = lr_scheduler.CosineAnnealingWithWarmup(cfg.epochs, 5)

    if cfg.bn_no_wd:
        def set_optimizer(trainer):
            bn_params = []
            non_bn_parameters = []
            for name, p in trainer.model.named_parameters():
                if "norm" in name:
                    bn_params.append(p)
                else:
                    non_bn_parameters.append(p)
            optim_params = [
                {"params": bn_params, "weight_decay": 0},
                {"params": non_bn_parameters, "weight_decay": cfg.weight_decay},
            ]
            trainer.optimizer = torch.optim.SGD(optim_params, lr=cfg.lr, momentum=0.9)

        trainers.SupervisedTrainer.set_optimizer = set_optimizer

    with trainers.SupervisedTrainer(model,
                                    optimizer,
                                    F.cross_entropy,
                                    reporters=[reporters.TensorboardReporter('.')],
                                    scheduler=scheduler,
                                    use_amp=cfg.use_amp,
                                    use_channel_last=cfg.use_channel_last,
                                    debug=cfg.debug
                                    ) as trainer:

        for _ in trainer.epoch_range(cfg.epochs):
            trainer.train(data.train_loader)
            trainer.test(data.test_loader)
            trainer.scheduler.step()

        print(f"Max Test Accuracy={max(trainer.reporter.history('accuracy/test')):.3f}")


if __name__ == '__main__':
    main()
