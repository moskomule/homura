import chika
import torch
from torch.nn import functional as F
from torchvision.models import resnet50

from homura import enable_accimage, get_num_nodes, init_distributed, lr_scheduler, optim, reporters
from homura.trainers import SupervisedTrainer
from homura.vision.data import DATASET_REGISTRY

is_distributed = False


def _detect_distributed():
    # handle raises Error with non key=val format
    # which causes problem with torch.distributed.launch
    import sys
    if any(['--local_rank' in k for k in sys.argv]):
        global is_distributed
        is_distributed = True


@chika.config
class Config:
    epochs: int = 90
    batch_size: int = 256
    enable_accimage: bool = False
    use_prefetcher: bool = False
    debug: bool = False
    use_amp: bool = False
    use_sync_bn: bool = False
    use_fast_collate: bool = False
    num_workers: 4

    init_method: str = "env://"
    backend: str = "nccl"


@chika.main(cfg_cls=Config)
def main(cfg: Config):
    if is_distributed:
        init_distributed(backend=cfg.backend,
                         init_method=cfg.init_method)
    if cfg.enable_accimage:
        enable_accimage()

    model = resnet50()
    optimizer = optim.SGD(lr=1e-1 * cfg.batch_size * get_num_nodes() / 256, momentum=0.9, weight_decay=1e-4)
    scheduler = lr_scheduler.MultiStepLR([30, 60, 80])
    train_loader, test_loader = DATASET_REGISTRY("fast_imagenet" if cfg.use_fast_collate else
                                                 "imagenet")(cfg.batch_size,
                                                             train_size=cfg.batch_size * 50 if cfg.debug else None,
                                                             test_size=cfg.batch_size * 50 if cfg.debug else None,
                                                             num_workers=cfg.num_workers,
                                                             use_prefetcher=cfg.use_prefetcher)

    use_multi_gpus = not is_distributed and torch.cuda.device_count() > 1
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
    import sys

    _detect_distributed()
    main()
