from pathlib import Path
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from homura.utils import reporter, callbacks, Trainer
from homura.models.vision.cifar import resnet20


def get_dataloader(batch_size, root="~/.torch/data/cifar10"):
    root = Path(root).expanduser()
    if not root.exists():
        root.mkdir()
    root = str(root)

    to_normalized_tensor = [transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    data_augmentation = [transforms.RandomCrop(32, padding=4),
                         transforms.RandomHorizontalFlip()]

    train_loader = DataLoader(
            datasets.CIFAR10(root, train=True, download=True,
                             transform=transforms.Compose(data_augmentation + to_normalized_tensor)),
            batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
            datasets.CIFAR10(root, train=False, transform=transforms.Compose(to_normalized_tensor)),
            batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


class ReporterCallback(callbacks.ReporterCallback):
    def end_epoch(self, data: dict):
        results = self.callback.end_epoch(data)
        self.reporter.add_scalars(results, "results", data["epoch"])


def main(batch_size):
    train_loader, test_loader = get_dataloader(batch_size)

    model = resnet20(num_classes=10)
    optimizer = optim.SGD(params=model.parameters(), lr=1e-1, momentum=0.9,
                          weight_decay=1e-4)
    c = callbacks.CallbackList(callbacks.AccuracyCallback(), callbacks.LossCallback())
    r = reporter.TQDMReporter(range(200))
    with ReporterCallback(r, c) as rep:
        trainer = Trainer(model, optimizer, F.cross_entropy, callbacks=rep)
        for _ in r:
            trainer.train(train_loader)
            trainer.test(test_loader)


if __name__ == '__main__':
    main(128)
