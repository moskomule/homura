from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50

from homura import optim, lr_scheduler
from homura.utils import callbacks, reporter, Trainer
from homura.vision.data import ImageFolder


def imagenet_loader(root, batch_size, num_workers=10, is_distributed=False, debug_mode=False):
    root = Path(root).expanduser()
    if not root.exists():
        raise FileNotFoundError

    _normalize = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    transform_base = [transforms.ToTensor(),
                      transforms.Normalize(*_normalize)]
    transfrom_train = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip()] + transform_base)
    transfrom_test = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224)] + transform_base)

    train_set = ImageFolder(root / "train", transform=transfrom_train, num_samples=840 if debug_mode else None)
    test_set = ImageFolder(root / "val", transform=transfrom_test, num_samples=840 if debug_mode else None)
    train_sampler = torch.utils.data.DistributedSampler(train_set) if is_distributed else None
    test_sampler = torch.utils.data.DistributedSampler(test_set) if is_distributed else None
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=train_sampler is None,
                              num_workers=num_workers, pin_memory=True,
                              sampler=train_sampler)
    test_loader = DataLoader(test_set,
                             batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
                             sampler=test_sampler)
    return train_loader, test_loader


def main(args, num_device):
    print(args)
    print("set model")
    model = resnet50()

    if args.distributed:
        from torch import distributed

        rank = args.local_rank
        torch.cuda.set_device(rank)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method=args.init_method)
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    else:
        rank = 0
        model = torch.nn.DataParallel(model, device_ids=list(range(num_device))).cuda()

    print("load data")
    train_loader, test_loader = imagenet_loader(args.root, args.batch_size, is_distributed=args.distributed)

    optimizer = optim.SGD(lr=1e-1 * num_device, momentum=0.9,
                          weight_decay=1e-4)
    scheduler = lr_scheduler.MultiStepLR([50, 70])

    if rank == 0:
        c = [callbacks.AccuracyCallback(), callbacks.LossCallback()]
        r = reporter.TQDMReporter(range(args.epochs), callbacks=c)
        tb = reporter.TensorboardReporter(c)
        rep = callbacks.CallbackList(r, tb, callbacks.WeightSave("checkpoints"))
    else:
        r = range(args.epochs)
        rep = None

    trainer = Trainer(model, optimizer, torch.nn.CrossEntropyLoss().cuda(), callbacks=rep,
                      scheduler=scheduler, use_cuda_nonblocking=True, verb=(rank == 0))
    for _ in r:
        trainer.train(train_loader)
        trainer.test(test_loader)


if __name__ == '__main__':
    import miniargs
    import warnings

    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

    p = miniargs.ArgumentParser()
    p.add_str("root")
    p.add_int("--epochs", default=90)
    p.add_int("--batch_size", default=128)
    p.add_true("--distributed")
    p.add_int("--local_rank", default=-1)
    p.add_str("--init_method", default="env://")
    p.add_true("--debug", help="Use less images and less epochs")
    args, _else = p.parse(return_unknown=True)
    num_device = torch.cuda.device_count()

    if not args.distributed and num_device <= 2:
        raise RuntimeError("requires multiple GPUs")
    if args.distributed and args.local_rank == -1:
        raise RuntimeError(
            f"For distributed training, use python -m torch.distributed.launch "
            f"--nproc_per_node={num_device} {__file__} {args.root} ...")
    main(args, num_device)
