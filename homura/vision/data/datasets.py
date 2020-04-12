import copy
import inspect
import pathlib
from dataclasses import dataclass
from typing import Optional, List, Callable

import numpy as np
import torch
from PIL import Image
from torchvision import datasets, transforms

from .utils import _Segmentation
from ..transforms import segmentation


# to enable _split_dataset
def _svhn_getitem(self,
                  index: int):
    img, target = self.data[index], int(self.targets[index])
    img = Image.fromarray(np.transpose(img, (1, 2, 0)))
    if self.transform is not None:
        img = self.transform(img)
    return img, target


datasets.SVHN.__getitem__ = _svhn_getitem


# Dataset(root, train, transform, download) is expected
class ImageNet(datasets.ImageNet):
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 download=False):
        assert not download, "Download dataset by yourself!"
        super(ImageNet, self).__init__(root, split="train" if train else "val", transform=transform)


class OriginalSVHN(datasets.SVHN):
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 download=False):
        super(OriginalSVHN, self).__init__(root, split="train" if train else "test", transform=transform,
                                           download=download)
        self.targets = self.labels


class ExtraSVHN(object):
    def __new__(cls,
                root,
                train=True,
                transform=None,
                download=False):
        if train:
            return (datasets.SVHN(root, split='train', transform=transform, download=download) +
                    datasets.SVHN(root, split='extra', transform=transform, download=download))
        else:
            return OriginalSVHN(root, train=False, transform=transform, download=download)


class ExtendedVOCSegmentation(object):
    def __new__(cls,
                root,
                train=True,
                transform=None,
                download=False):
        if train:
            return datasets.SBDataset(root, image_set='train_noval', mode='segmentation', download=download,
                                      transforms=transform)
        else:
            return datasets.VOCSegmentation(root, image_set='val', download=download, transforms=transform)


def _split_dataset(train_set: datasets.VisionDataset,
                   val_size: int) -> (datasets.VisionDataset, datasets.VisionDataset):
    # split train_set to train_set and val_set
    assert len(train_set) >= val_size
    indices = torch.randperm(len(train_set))
    valset = copy.deepcopy(train_set)
    train_set.data = [train_set.data[i] for i in indices[val_size:]]
    train_set.targets = [train_set.targets[i] for i in indices[val_size:]]

    valset.data = [valset.data[i] for i in indices[:val_size]]
    valset.targets = [valset.targets[i] for i in indices[:val_size]]

    return train_set, valset


@dataclass
class VisionSet:
    tv_class: type(datasets.VisionDataset)
    root: str or pathlib.Path
    num_classes: int
    default_norm: List
    default_train_da: Optional[List] = None
    default_test_da: Optional[List] = None
    collate_fn: Optional[Callable] = None

    def __post_init__(self):
        # _ is self
        _, *args = inspect.getfullargspec(self.tv_class).args
        assert {'root', 'train', 'transform', 'download'} < set(args), \
            "dataset DataSet(root, train, transform) is expected"
        self.root = pathlib.Path(self.root).expanduser()
        if self.default_train_da is None:
            self.default_train_da = []
        if self.default_test_da is None:
            self.default_test_da = []

    def instantiate(self,
                    train_da: Optional[List] = None,
                    test_da: Optional[List] = None,
                    norm: Optional[List] = None,
                    download: bool = False
                    ) -> (datasets.VisionDataset, datasets.VisionDataset):
        assert (download or self.root.exists()), "root does not exist"
        if train_da is None:
            train_da = self.default_train_da
        if test_da is None:
            test_da = self.default_test_da
        if norm is None:
            norm = self.default_norm
        train_transform = transforms.Compose(train_da + norm)
        train_set = self.tv_class(self.root, train=True, transform=train_transform, download=download)
        test_transform = transforms.Compose(test_da + norm)
        test_set = self.tv_class(self.root, train=False, transform=test_transform, download=download)
        return train_set, test_set


_DATASETS = {'cifar10': VisionSet(datasets.CIFAR10, "~/.torch/data/cifar10", 10,
                                  [transforms.ToTensor(),
                                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))],
                                  [transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                                   transforms.RandomHorizontalFlip()]),

             'cifar100': VisionSet(datasets.CIFAR100, "~/.torch/data/cifar100", 100,
                                   [transforms.ToTensor(),
                                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))],
                                   [transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                                    transforms.RandomHorizontalFlip()]),

             'svhn': VisionSet(OriginalSVHN, "~/.torch/data/svhn", 10,
                               [transforms.ToTensor(),
                                transforms.Normalize((0.4390, 0.4443, 0.4692), (0.1189, 0.1222, 0.1049))],
                               [transforms.RandomCrop(32, padding=4, padding_mode='reflect')]),

             'extra_svhn': VisionSet(ExtraSVHN, "~/.torch/data/svhn", 10,
                                     [transforms.ToTensor(),
                                      transforms.Normalize((0.4390, 0.4443, 0.4692), (0.1189, 0.1222, 0.1049))],
                                     [transforms.RandomCrop(32, padding=4, padding_mode='reflect')]),

             'mnist': VisionSet(datasets.MNIST, "~/.torch/data/mnist", 10,
                                [transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))],
                                []),

             'kmnist': VisionSet(datasets.KMNIST, "~/.torch/data/kmnist", 10,
                                 [transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))],
                                 []),

             'imagenet': VisionSet(ImageNet, "~/.torch/data/imagenet", 1_000,
                                   [transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))],
                                   [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()],
                                   [transforms.Resize(256), transforms.CenterCrop(224)]),

             'voc_seg': VisionSet(ExtendedVOCSegmentation, "~/.torch/data/voc", 21,
                                  [segmentation.ToTensor(),
                                   segmentation.Normalize((0.3265, 0.3116, 0.2888, (0.2906, 0.2815, 0.2753)))],
                                  [segmentation.RandomResize(260, 1040),
                                   segmentation.RandomHorizontalFlip(),
                                   segmentation.RandomCrop(480)],
                                  default_test_da=[segmentation.RandomResize(520)],
                                  collate_fn=_Segmentation.collate_fn)
             }


def register_vision_dataset(name: str,
                            vision_set: VisionSet
                            ) -> None:
    if name in _DATASETS.keys():
        raise RuntimeError(f'name=({name}) is already registered')
    if isinstance(vision_set, VisionSet):
        raise RuntimeError(f'vision_set is expected to be VisionSet, '
                           f'but got {type(vision_set)} instead.')
    _DATASETS[name] = vision_set
