from pathlib import Path

import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import densenet121

from homura.utils import callbacks, reporter, Trainer
from homura.data import ImageFolder


def imagenet_loader(root, batch_size, num_workers=16):
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

    train_set = ImageFolder(root / "train", transform=transfrom_train, on_memory=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(ImageFolder(root / "val", transform=transfrom_test, on_memory=True),
                             batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def main(root, epochs, batch_size):
    train_loader, test_loader = imagenet_loader(root, batch_size)
    model = torch.nn.DataParallel(densenet121(), device_ids=(0, 1, 2, 3))
    optimizer = optim.SGD(params=model.parameters(), lr=1e-1, momentum=0.9,
                          weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [50, 70])
    c = callbacks.CallbackList(callbacks.AccuracyCallback(), callbacks.LossCallback(),
                               callbacks.WeightSave("checkpoints"))
    r = reporter.TQDMReporter(range(epochs))
    with callbacks.ReporterCallback(r, c) as rep:
        trainer = Trainer(model, optimizer, F.cross_entropy, callbacks=rep,
                          scheduler=scheduler)
        for _ in r:
            trainer.train(train_loader)
            trainer.test(test_loader)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--epochs", default=1)
    p.add_argument("--batch_size", default=128)
    args = p.parse_args()
    main(**vars(args))
