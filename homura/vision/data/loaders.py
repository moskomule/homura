from pathlib import Path

from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from torchvision import datasets, transforms

from .folder import ImageFolder


class _BaseLoaders(object):
    def __init__(self, dataset, mean_std: tuple,
                 data_augmentation_transforms: list,
                 test_time_transforms: list = None,
                 replacement: bool = False, distributed: bool = False):
        self._dataset = dataset

        self._da_transform = [] if data_augmentation_transforms is None else data_augmentation_transforms
        self._tt_transform = [] if test_time_transforms is None else test_time_transforms
        self._norm_transform = [transforms.ToTensor(),
                                transforms.Normalize(*mean_std)]
        self._distributed = distributed
        self._replacement = replacement

    def __call__(self, batch_size: int, num_workers: int, shuffle: bool, train_set_kwargs: dict, test_set_kwargs: dict):
        shuffle = (not self._distributed) and shuffle
        train_set = self._dataset(**train_set_kwargs,
                                  transform=transforms.Compose(self._da_transform + self._norm_transform))
        test_set = self._dataset(**test_set_kwargs,
                                 transform=transforms.Compose(self._tt_transform + self._norm_transform))
        train_sampler, test_sampler = None, None
        if self._distributed:
            train_sampler = DistributedSampler(train_set)
            test_sampler = DistributedSampler(test_set)
        elif self._replacement:
            train_sampler = RandomSampler(train_set, replacement=True, num_samples=len(train_set) // batch_size)
        train = DataLoader(train_set, sampler=train_sampler,
                           batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=True)
        test = DataLoader(test_set, sampler=test_sampler,
                          batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=True)
        return train, test

    @staticmethod
    def absolute_root(root):
        root = Path(root).expanduser()
        if not root.exists():
            root.mkdir(parents=True)
        return str(root)

    @staticmethod
    def check_root_exists(root):
        root = Path(root).expanduser()
        if not root.exists():
            raise FileNotFoundError(f"Cannot find {root}")
        return root


def cifar10_loaders(batch_size, num_workers=1, root="~/.torch/data/cifar10", data_augmentation=None):
    if data_augmentation is None:
        data_augmentation = [transforms.RandomCrop(32, padding=4),
                             transforms.RandomHorizontalFlip()]
    _base = _BaseLoaders(datasets.CIFAR10, ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), data_augmentation)

    root = _BaseLoaders.absolute_root(root)
    train_loader, test_loader = _base(batch_size=batch_size, num_workers=num_workers, shuffle=True,
                                      train_set_kwargs=dict(root=root, train=True, download=True),
                                      test_set_kwargs=dict(root=root, train=False, download=True))

    return train_loader, test_loader


def cifar100_loaders(batch_size, num_workers=1, root="~/.torch/data/cifar100", data_augmentation=None):
    if data_augmentation is None:
        data_augmentation = [transforms.RandomCrop(32, padding=4),
                             transforms.RandomHorizontalFlip()]
    _base = _BaseLoaders(datasets.CIFAR10, ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)), data_augmentation)

    root = _BaseLoaders.absolute_root(root)
    train_loader, test_loader = _base(batch_size=batch_size, num_workers=num_workers, shuffle=True,
                                      train_set_kwargs=dict(root=root, train=True, download=True),
                                      test_set_kwargs=dict(root=root, train=False, download=True))

    return train_loader, test_loader


def imagenet_loaders(root, batch_size, num_workers=8, data_augmentation=None, num_train_samples=None,
                     num_test_samples=None, distributed=False):
    import torch

    if distributed:
        batch_size = batch_size // torch.cuda.device_count()
    if data_augmentation is None:
        data_augmentation = [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]
    tt_transform = [transforms.Resize(256), transforms.CenterCrop(224)]
    _base = _BaseLoaders(ImageFolder, ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), data_augmentation, tt_transform,
                         distributed=distributed)
    root = _BaseLoaders.check_root_exists(root)
    train_loader, test_loader = _base(batch_size=batch_size, num_workers=num_workers, shuffle=True,
                                      train_set_kwargs=dict(root=(root / "train"), num_samples=num_train_samples),
                                      test_set_kwargs=dict(root=(root / "val"), num_samples=num_test_samples))
    return train_loader, test_loader
