import torch
from torch.nn import functional as F
from torchvision.models import resnet50

from homura import optim, lr_scheduler, callbacks, reporters, enable_accimage, get_num_nodes, init_distributed
from homura.trainers import SupervisedTrainer
from homura.vision.data import imagenet_loaders, prefetcher


def main():
    if args.distributed:
        init_distributed(use_horovod=args.use_horovod,
                         backend=args.backend,
                         init_method=args.init_method)
    if args.enable_accimage:
        enable_accimage()

    model = resnet50()
    optimizer = optim.SGD(lr=1e-1 * args.batch_size * get_num_nodes() / 256, momentum=0.9, weight_decay=1e-4)
    scheduler = lr_scheduler.MultiStepLR([30, 60, 80])
    tq = reporters.TQDMReporter(range(args.epochs))
    c = [callbacks.AccuracyCallback(),
         callbacks.AccuracyCallback(k=5),
         callbacks.LossCallback(),
         tq,
         reporters.TensorboardReporter("."),
         reporters.IOReporter(".")]
    _train_loader, _test_loader = imagenet_loaders(args.root,
                                                   args.batch_size,
                                                   distributed=args.distributed,
                                                   num_train_samples=args.batch_size * 10 if args.debug else None,
                                                   num_test_samples=args.batch_size * 10 if args.debug else None)

    use_multi_gpus = not args.distributed and torch.cuda.device_count() > 1
    with SupervisedTrainer(model,
                           optimizer,
                           F.cross_entropy,
                           callbacks=c,
                           scheduler=scheduler,
                           data_parallel=use_multi_gpus,
                           use_horovod=args.use_horovod) as trainer:

        for epoch in tq:
            if args.use_prefetcher:
                train_loader = prefetcher.DataPrefetcher(_train_loader)
                test_loader = prefetcher.DataPrefetcher(_test_loader)
            else:
                train_loader, test_loader = _train_loader, _test_loader
            # following apex's training scheme
            trainer.train(train_loader)
            trainer.test(test_loader)


if __name__ == '__main__':
    import miniargs
    import warnings

    # to suppress annoying warnings from PIL
    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

    p = miniargs.ArgumentParser()
    p.add_str("root")
    p.add_true("--use_horovod")
    p.add_int("--epochs", default=90)
    p.add_int("--batch_size", default=256)
    p.add_true("--distributed")
    p.add_int("--local_rank", default=-1)
    p.add_str("--init_method", default="env://")
    p.add_str("--backend", default="nccl")
    p.add_true("--enable_accimage")
    p.add_true("--use_prefetcher")
    p.add_true("--debug", help="Use less images and less epochs")
    args, _else = p.parse(return_unknown=True)
    args.distributed = args.use_horovod or args.distributed

    print(args)
    main()
