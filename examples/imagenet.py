import chika
import torch
from torch.nn import functional as F
from torchvision.models import resnet50

from homura import distributed_ready_main, enable_accimage, get_num_nodes, is_distributed, lr_scheduler, optim, \
    reporters
from homura.trainers import SupervisedTrainer
from homura.vision.data import DATASET_REGISTRY


@chika.config
class Config:
    epochs: int = 90
    batch_size: int = 256
    enable_accimage: bool = False
    debug: bool = False
    use_amp: bool = False
    use_sync_bn: bool = False
    num_workers: int = 4

    init_method: str = "env://"
    backend: str = "nccl"


@chika.main(cfg_cls=Config)
@distributed_ready_main
def main(cfg: Config):
    if cfg.enable_accimage:
        enable_accimage()

    model = resnet50()
    optimizer = optim.SGD(lr=1e-1 * cfg.batch_size * get_num_nodes() / 256, momentum=0.9, weight_decay=1e-4)
    scheduler = lr_scheduler.MultiStepLR([30, 60, 80])
    train_loader, test_loader = DATASET_REGISTRY("fast_imagenet" if cfg.use_fast_collate else
                                                 "imagenet")(cfg.batch_size,
                                                             train_size=cfg.batch_size * 50 if cfg.debug else None,
                                                             test_size=cfg.batch_size * 50 if cfg.debug else None,
                                                             num_workers=cfg.num_workers)

    use_multi_gpus = not is_distributed() and torch.cuda.device_count() > 1
    with SupervisedTrainer(model,
                           optimizer,
                           F.cross_entropy,
                           reporters=[reporters.TensorboardReporter(".")],
                           scheduler=scheduler,
                           data_parallel=use_multi_gpus,
                           use_amp=cfg.use_amp,
                           use_cuda_nonblocking=True,
                           use_sync_bn=cfg.use_sync_bn,
                           report_accuracy_topk=5) as trainer:

        for epoch in trainer.epoch_range(cfg.epochs):
            trainer.train(train_loader)
            trainer.test(test_loader)

        print(f"Max Test Accuracy={max(trainer.reporter.history('accuracy/test')):.3f}")


if __name__ == '__main__':
    import warnings

    # to suppress annoying warnings from PIL
    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

    main()
