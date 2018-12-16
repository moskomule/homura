from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class _BaseLoaders(object):
    def __init__(self, dataset,
                 data_augmentation_transform: list, mean_std: tuple):
        self._dataset = dataset

        self._da_transform = data_augmentation_transform
        self._norm_transform = [transforms.ToTensor(),
                                transforms.Normalize(*mean_std)]

    def __call__(self, batch_size, num_workers, shuffle, train_set_kwargs, test_set_kwargs):
        train = DataLoader(self._dataset(**train_set_kwargs, train=True, download=True,
                                         transform=transforms.Compose(self._da_transform + self._norm_transform)),
                           batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
        test = DataLoader(self._dataset(**test_set_kwargs, train=False, download=True,
                                        transform=transforms.Compose(self._norm_transform)),
                          batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
        return train, test

    @staticmethod
    def absolute_root(root):
        root = Path(root).expanduser()
        if not root.exists():
            root.mkdir(parents=True)
        return str(root)


def cifar10_loaders(batch_size, num_workers=1, root="~/.torch/data/cifar10", data_augmentation=None):
    if data_augmentation is None:
        data_augmentation = [transforms.RandomCrop(32, padding=4),
                             transforms.RandomHorizontalFlip()]
    _base = _BaseLoaders(datasets.CIFAR10, data_augmentation,
                         ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))

    root = _BaseLoaders.absolute_root(root)
    train_loader, test_loader = _base(batch_size=batch_size, num_workers=num_workers, shuffle=True,
                                      train_set_kwargs=dict(root=root), test_set_kwargs=dict(root=root))

    return train_loader, test_loader


def cifar100_loaders(batch_size, num_workers=1, root="~/.torch/data/cifar100", data_augmentation=None):
    if data_augmentation is None:
        data_augmentation = [transforms.RandomCrop(32, padding=4),
                             transforms.RandomHorizontalFlip()]
    _base = _BaseLoaders(datasets.CIFAR10, data_augmentation,
                         ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)))

    root = _BaseLoaders.absolute_root(root)
    train_loader, test_loader = _base(batch_size=batch_size, num_workers=num_workers, shuffle=True,
                                      train_set_kwargs=dict(root=root), test_set_kwargs=dict(root=root))

    return train_loader, test_loader
