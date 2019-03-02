from pathlib import Path
from typing import Iterable, Union, Optional

from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from torchvision import datasets, transforms

from .custom_dataset import transformable_random_split
from .folder import ImageFolder


class BaseLoaders(object):
    """ A base class for simple data-loaders

    :param dataset:
    :param mean_std:
    :param data_augmentation_transforms:
    :param test_time_transforms:
    :param replacement:
    :param num_train_samples:
    :param distributed:
    """

    def __init__(self, dataset,
                 mean_std: tuple,
                 data_augmentation_transforms: list,
                 test_time_transforms: list = None,
                 replacement: bool = False,
                 num_train_samples: Optional[int] = None,
                 distributed: bool = False):

        self._dataset = dataset

        self._da_transform = [] if data_augmentation_transforms is None else data_augmentation_transforms
        self._tt_transform = [] if test_time_transforms is None else test_time_transforms
        self._norm_transform = [transforms.ToTensor(),
                                transforms.Normalize(*mean_std)]
        self._replacement = replacement
        if self._replacement and num_train_samples is None:
            raise RuntimeError("num_train_samples should be specified if `replacement=True`")
        self._num_train_samples = num_train_samples
        self._distributed = distributed

    def __call__(self, batch_size: int,
                 num_workers: int,
                 shuffle: bool,
                 train_set_kwargs: dict,
                 test_set_kwargs: dict,
                 val_size: int = 0):
        """

        :param batch_size:
        :param num_workers:
        :param shuffle:
        :param train_set_kwargs:
        :param test_set_kwargs:
        :param val_size:
        :return:
        """

        shuffle = (not self._distributed) and shuffle

        test_transform = transforms.Compose(self._tt_transform + self._norm_transform)
        if val_size == 0:
            train_transform = transforms.Compose(self._da_transform + self._norm_transform)
        else:
            # if val_size > 0
            train_transform = None
        train_set = self._dataset(**train_set_kwargs, transform=train_transform)
        test_set = self._dataset(**test_set_kwargs,
                                 transform=test_transform)
        if val_size > 0:
            # split and transform
            train_set, val_set = transformable_random_split(train_set, [len(train_set) - val_size, val_size])
            train_set.update_transforms(transforms.Compose(self._da_transform + self._norm_transform))
            val_set.update_transforms(test_transform)

        train_sampler, test_sampler, val_sampler = None, None, None
        if self._distributed:
            train_sampler = DistributedSampler(train_set)
            test_sampler = DistributedSampler(test_set)
            if val_size > 0:
                val_sampler = DistributedSampler(val_set)

        elif self._replacement:
            if self._num_train_samples > len(train_set):
                raise RuntimeWarning(f"Required number of train samples {self._num_train_samples} is "
                                     f"larger than the size of train set {len(train_set)}."
                                     f"It works but may cause problem!")
            train_sampler = RandomSampler(train_set, replacement=True, num_samples=self._num_train_samples)

        train = DataLoader(train_set, sampler=train_sampler,
                           batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=True)
        test = DataLoader(test_set, sampler=test_sampler,
                          batch_size=2 * batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
        if val_size == 0:
            return train, test

        val = DataLoader(val_set, sampler=val_sampler, batch_size=2 * batch_size, num_workers=num_workers,
                         shuffle=False, pin_memory=True)
        return train, test, val

    @staticmethod
    def absolute_root(root):
        root = Path(root).expanduser()
        return root

    @staticmethod
    def check_root_exists(root):
        root = Path(root).expanduser()
        if not root.exists():
            raise FileNotFoundError(f"Cannot find {root}")
        return root


def _base_mnists_loaders(cls,
                         batch_size: int,
                         num_workers: int = 2,
                         root: str = "~/.torch/data/mnist",
                         data_augmentation: Optional[Iterable] = None,
                         val_size: int = 0,
                         replacement: bool = False,
                         num_train_samples: Optional[int] = None,
                         force_download: bool = False,
                         mean_and_std: tuple = ((0.1307,), (0.3081,))):
    if data_augmentation is None:
        data_augmentation = [transforms.RandomHorizontalFlip()]
    _base = BaseLoaders(cls, mean_and_std, data_augmentation, replacement=replacement,
                        num_train_samples=num_train_samples)
    root = BaseLoaders.absolute_root(root)
    return _base(batch_size=batch_size, num_workers=num_workers, shuffle=not replacement, val_size=val_size,
                 train_set_kwargs=dict(root=root, train=True,
                                       download=not root.exists() or force_download),
                 test_set_kwargs=dict(root=root, train=False,
                                      download=not root.exists() or force_download))


def mnist_loaders(batch_size: int,
                  num_workers: int = 2,
                  root: str = "~/.torch/data/mnist",
                  data_augmentation: Iterable = None,
                  val_size: int = 0,
                  replacement: bool = False,
                  num_train_samples: Optional[int] = None,
                  force_download: bool = False):
    """ A simple data loader for MNIST.

    :param batch_size:
    :param num_workers:
    :param root:
    :param data_augmentation: default transformation is RandomHorizontalFlip()
    :param val_size:
    :param replacement: sampling with replacement
    :param num_train_samples: number of data sampled when `replacement` is True
    :param force_download:
    :return: (train_loader, test_loader) if `val_size==0` else (train_loader, test_loader, val_loader)
    """

    return _base_mnists_loaders(datasets.MNIST, batch_size, num_workers, root, data_augmentation, val_size, replacement,
                                num_train_samples, force_download)


def fashion_mnist_loaders(batch_size: int,
                          num_workers: int = 2,
                          root: str = "~/.torch/data/fmnist",
                          data_augmentation: Optional[Iterable] = None,
                          val_size: int = 0,
                          replacement: bool = False,
                          num_train_samples: Optional[int] = None,
                          force_download: bool = False):
    # A simple loaders for Fashion MNIST a.k.a. FMNIST
    return _base_mnists_loaders(datasets.FashionMNIST, batch_size, num_workers, root, data_augmentation, val_size,
                                replacement, num_train_samples, force_download)


def kuzushiji_mnist_loaders(batch_size: int,
                            num_workers: int = 2,
                            root: str = "~/.torch/data/kmnist",
                            data_augmentation: Optional[Iterable] = None,
                            val_size: int = 0,
                            replacement: bool = False,
                            num_train_samples: Optional[int] = None,
                            force_download: bool = False):
    # A simple loaders for Kuzushiji MNIST a.k.a. KMINIST
    return _base_mnists_loaders(datasets.KMNIST, batch_size, num_workers, root, data_augmentation, val_size,
                                replacement, num_train_samples, force_download)


def cifar10_loaders(batch_size: int,
                    num_workers: int = 4,
                    root: str = "~/.torch/data/cifar10",
                    data_augmentation: Optional[Iterable] = None,
                    val_size: int = 0,
                    replacement: bool = False,
                    num_train_samples: Optional[int] = None,
                    force_download: bool = False):
    """ A simple data loader for CIFAR-10

    :param batch_size:
    :param num_workers:
    :param root:
    :param data_augmentation: default transformation is RandomCrop(32, padding=4) and RandomHorizontalFlip()
    :param val_size:
    :param replacement: sampling with replacement
    :param num_train_samples: number of data sampled when `replacement` is True
    :param force_download:
    :return: (train_loader, test_loader) if `val_size==0` else (train_loader, test_loader, val_loader)
    """
    if data_augmentation is None:
        data_augmentation = [transforms.RandomCrop(32, padding=4),
                             transforms.RandomHorizontalFlip()]
    _base = BaseLoaders(datasets.CIFAR10, ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), data_augmentation,
                        replacement=replacement, num_train_samples=num_train_samples)

    root = BaseLoaders.absolute_root(root)
    return _base(batch_size=batch_size, num_workers=num_workers, shuffle=not replacement, val_size=val_size,
                 train_set_kwargs=dict(root=root, train=True,
                                       download=not root.exists() or force_download),
                 test_set_kwargs=dict(root=root, train=False,
                                      download=not root.exists() or force_download))


def cifar100_loaders(batch_size: int,
                     num_workers: int = 4,
                     root: str = "~/.torch/data/cifar100",
                     data_augmentation: Optional[Iterable] = None,
                     val_size: int = 0,
                     replacement: bool = False,
                     num_train_samples: Optional[int] = None,
                     force_download: bool = False):
    """ A simple data loader for CIFAR-100

    :param batch_size:
    :param num_workers:
    :param root:
    :param data_augmentation: default transformation is `RandomCrop(32, padding=4)` and `RandomHorizontalFlip()`
    :param val_size:
    :param replacement: sampling with replacement
    :param num_train_samples: number of data sampled when `replacement` is True
    :param force_download:
    :return: (train_loader, test_loader) if `val_size==0` else (train_loader, test_loader, val_loader)
    """
    if data_augmentation is None:
        data_augmentation = [transforms.RandomCrop(32, padding=4),
                             transforms.RandomHorizontalFlip()]
    _base = BaseLoaders(datasets.CIFAR100, ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)), data_augmentation,
                        replacement=replacement, num_train_samples=num_train_samples)

    root = BaseLoaders.absolute_root(root)
    return _base(batch_size=batch_size, num_workers=num_workers, shuffle=not replacement, val_size=val_size,
                 train_set_kwargs=dict(root=root, train=True,
                                       download=not root.exists() or force_download),
                 test_set_kwargs=dict(root=root, train=False,
                                      download=not root.exists() or force_download))


def imagenet_loaders(root: Union[Path, str],
                     batch_size: int,
                     num_workers: int = 8,
                     data_augmentation: Optional[Iterable] = None,
                     val_size: int = 0,
                     num_train_samples: Optional[int] = None,
                     num_test_samples: Optional[int] = None,
                     distributed: bool = False):
    """ A simple data loader for ILSVRC classification data set

    :param root:
    :param batch_size: Total batch size
    :param num_workers:
    :param data_augmentation:
    :param val_size:
    :param num_train_samples:
    :param num_test_samples:
    :param distributed:
    :return: tuple of train and test loaders
    """
    import torch

    if distributed:
        batch_size = batch_size // torch.cuda.device_count()
    if data_augmentation is None:
        data_augmentation = [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]
    tt_transform = [transforms.Resize(256), transforms.CenterCrop(224)]
    _base = BaseLoaders(ImageFolder, ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), data_augmentation, tt_transform,
                        distributed=distributed)
    root = BaseLoaders.check_root_exists(root)
    return _base(batch_size=batch_size, num_workers=num_workers, shuffle=True, val_size=val_size,
                 train_set_kwargs=dict(root=(root / "train"), num_samples=num_train_samples),
                 test_set_kwargs=dict(root=(root / "val"), num_samples=num_test_samples))
