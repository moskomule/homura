from pathlib import Path

import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from homura import optim, lr_scheduler
from homura.utils import reporter, callbacks, Trainer
from homura.vision.models.cifar import resnet20
from homura.vision.transforms import RandomErase


def get_dataloader(batch_size, root="~/.torch/data/cifar10"):
    root = Path(root).expanduser()
    if not root.exists():
        root.mkdir(parents=True)
    root = str(root)

    to_normalized_tensor = [transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    data_augmentation = [transforms.RandomCrop(32, padding=4),
                         RandomErase(0.2, 0.1, 1),
                         transforms.RandomHorizontalFlip()]

    train_loader = DataLoader(
        datasets.CIFAR10(root, train=True, download=True,
                         transform=transforms.Compose(data_augmentation + to_normalized_tensor)),
        batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        datasets.CIFAR10(root, train=False, transform=transforms.Compose(to_normalized_tensor)),
        batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def main(batch_size):
    train_loader, test_loader = get_dataloader(batch_size)

    model = resnet20(num_classes=10)
    optimizer = optim.SGD(lr=1e-1, momentum=0.9, weight_decay=1e-4)
    scheduler = lr_scheduler.MultiStepLR([100, 150])
    c = [callbacks.AccuracyCallback(), callbacks.LossCallback()]
    r = reporter.TQDMReporter(range(200), callbacks=c)
    tb = reporter.TensorboardReporter(c)
    tb.report_parameters()

    with callbacks.CallbackList(r, tb) as rep:
        trainer = Trainer(model, optimizer, F.cross_entropy, callbacks=rep, scheduler=scheduler)
        for _ in r:
            trainer.train(train_loader)
            trainer.test(test_loader)


if __name__ == '__main__':
    main(128)
