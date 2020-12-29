import copy
import inspect
import pathlib
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Tuple, Type

import numpy as np
import torch
from PIL import Image
from torch.jit.annotations import List
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from torchvision import datasets, transforms

from homura import Registry, get_environ, is_distributed
from .prefetcher import DataPrefetchWrapper


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
class ImageNet(datasets.ImageFolder):
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 download=False):
        assert not download, "Download dataset by yourself!"
        root = pathlib.Path(root) / ('train' if train else 'val')
        super(ImageNet, self).__init__(root, transform=transform)
        import warnings

        warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


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


def fast_collate(batch: List[Tuple[Image.Image, int]],
                 memory_format: torch.memory_format,
                 mean: Tuple[int, ...],
                 std: Tuple[int, ...]
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.long)
    mean = torch.as_tensor(mean, dtype=torch.float)
    std = torch.as_tensor(std, dtype=torch.float)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.empty((len(imgs), 3, h, w), dtype=torch.uint8).contiguous(memory_format=memory_format)
    for i, img in enumerate(imgs):
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        img = img.view(h, w, 3).permute((2, 0, 1)).contiguous()
        tensor[i].copy_(img)
    tensor = tensor.float().div_(255)
    tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
    return tensor, targets


@dataclass
class VisionSet:
    """ Dataset abstraction for vision datasets.
    """

    tv_class: Type[datasets.VisionDataset]
    root: str or pathlib.Path
    num_classes: int
    default_norm: List
    default_train_da: Optional[List] = None
    default_test_da: Optional[List] = None
    collate_fn: Optional[Callable] = None

    def __post_init__(self):
        # _ is trainer
        _, *args = inspect.getfullargspec(self.tv_class.__init__).args
        if not ({'root', 'train', 'transform', 'download'} <= set(args)):
            raise RuntimeError(f"dataset DataSet(root, train, transform, download) is expected, "
                               f"but {self.tv_class} has arguments of {args} instead.")
        self.root = pathlib.Path(self.root).expanduser()
        if self.default_train_da is None:
            self.default_train_da = []
        if self.default_test_da is None:
            self.default_test_da = []

    def get_dataset(self,
                    train_size: Optional[int] = None,
                    test_size: Optional[int] = None,
                    val_size: Optional[int] = None,
                    train_da: Optional[List] = None,
                    test_da: Optional[List] = None,
                    norm: Optional[List] = None,
                    download: bool = False,
                    *,
                    pre_default_train_da: Optional[List] = None,
                    post_default_train_da: Optional[List] = None,
                    post_norm_train_da: Optional[List] = None
                    ) -> Tuple[datasets.VisionDataset, datasets.VisionDataset, Optional[datasets.VisionDataset]]:
        """ Get Dataset

        :param train_size: Size of training dataset. If None, full dataset will be available.
        :param test_size: Size of test dataset. If None, full dataset will be available.
        :param val_size: Size of validation dataset, randomly split from the training dataset. If None, None will be returned.
        :param train_da: Data Augmentation for training
        :param test_da: Data Augmentation for testing and validation
        :param norm: Normalization after train_da and test_da
        :param download: If dataset needs downloading
        :param pre_default_train_da: Data Augmentation before the default data augmentation
        :param post_default_train_da: Data Augmentation after the default data augmentation
        :param post_norm_train_da: Data Augmentation after normalization (i.e., norm)
        :return: train_set, test_set, Optional[val_set]
        """

        assert (download or self.root.exists()), "root does not exist"
        if train_da is None:
            train_da = list(self.default_train_da)
        if test_da is None:
            test_da = list(self.default_test_da)
        if norm is None:
            norm = list(self.default_norm)

        def unpack_optional_list(x: Optional[List]) -> List:
            return [] if x is None else x

        pre_default_train_da = unpack_optional_list(pre_default_train_da)
        post_default_train_da = unpack_optional_list(post_default_train_da)
        post_norm_train_da = unpack_optional_list(post_norm_train_da)

        train_transform = transforms.Compose(pre_default_train_da + train_da + post_default_train_da
                                             + norm + post_norm_train_da)
        train_set = self.tv_class(self.root, train=True, transform=train_transform, download=download)
        if train_size is not None and train_size > len(train_set):
            raise ValueError(f'train_size should be <={len(train_set)}')

        test_transform = transforms.Compose(test_da + norm)
        test_set = self.tv_class(self.root, train=False, transform=test_transform, download=download)
        if test_size is not None and test_size > len(test_set):
            raise ValueError(f'test_size should be <={len(test_set)}')

        val_set = None
        if val_size is not None and val_size > 0:
            if train_size is not None and (train_size + val_size) > len(train_set):
                raise ValueError(f'train_set+val_size should be <={len(train_set)}')

            train_set, val_set = self._split_dataset(train_set, val_size)
            val_set.transform = test_transform

        if train_size is not None:
            train_set = self._sample_dataset(train_set, train_size)

        if test_size is not None:
            test_set = self._sample_dataset(test_set, test_size)

        return train_set, test_set, val_set

    def get_dataloader(self,
                       batch_size: int,
                       train_da: Optional[List] = None,
                       test_da: Optional[List] = None,
                       norm: Optional[List] = None,
                       train_size: Optional[int] = None,
                       test_size: Optional[int] = None,
                       val_size: Optional[int] = None,
                       download: bool = False,
                       num_workers: int = 1,
                       non_training_bs_factor=2,
                       drop_last: bool = False,
                       pin_memory: bool = True,
                       return_num_classes: bool = False,
                       test_batch_size: Optional[int] = None,
                       pre_default_train_da: Optional[List] = None,
                       post_default_train_da: Optional[List] = None,
                       post_norm_train_da: Optional[List] = None,
                       use_prefetcher: bool = False,
                       start_epoch: bool = 0
                       ) -> (Tuple[DataLoader, DataLoader]
                             or Tuple[DataLoader, DataLoader, DataLoader]
                             or Tuple[DataLoader, DataLoader, int]
                             or Tuple[DataLoader, DataLoader, DataLoader, int]):
        """ Get Dataloader. This will automatically handle distributed setting

        :param batch_size: Batch size
        :param train_da: Data Augmentation for training
        :param test_da: Data Augmentation for testing and validation
        :param norm: Normalization after train_da and test_da
        :param train_size: Size of training dataset. If None, full dataset will be available.
        :param test_size: Size of test dataset. If None, full dataset will be available.
        :param val_size: Size of validation dataset, randomly split from the training dataset.
        If None, None will be returned.
        :param download: If dataset needs downloading
        :param num_workers: Number of workers in data loaders
        :param non_training_bs_factor: Batch size scale factor during non training. For example,
        testing time requires no backward cache, so basically batch size can be doubled.
        :param drop_last: If drop last batch or not
        :param pin_memory: If pin memory or not
        :param return_num_classes: If return number of classes as the last return value
        :param test_batch_size: Test time batch size. If None, non_training_bs_factor * batch_size is used.
        :param pre_default_train_da: Data Augmentation before the default data augmentation
        :param post_default_train_da: Data Augmentation after the default data augmentation
        :param post_norm_train_da: Data Augmentation after normalization (i.e., norm)
        :param use_prefetcher: Use prefetcher or Not
        :param start_epoch: Epoch at start time
        :return: train_loader, test_loader, [val_loader], [num_classes]
        """

        train_set, test_set, val_set = self.get_dataset(train_size, test_size, val_size,
                                                        train_da, test_da, norm, download,
                                                        pre_default_train_da=pre_default_train_da,
                                                        post_default_train_da=post_default_train_da,
                                                        post_norm_train_da=post_norm_train_da)
        if test_batch_size is None:
            test_batch_size = non_training_bs_factor * batch_size

        samplers = [None, None, None]
        if is_distributed():
            import homura

            dist_sampler_kwargs = dict(num_replicas=homura.get_world_size(),
                                       rank=homura.get_global_rank())
            samplers[0] = DistributedSampler(train_set, **dist_sampler_kwargs)
            samplers[2] = DistributedSampler(test_set, **dist_sampler_kwargs)
            samplers[0].set_epoch(start_epoch)
            samplers[2].set_epoch(start_epoch)
        else:
            samplers[0] = RandomSampler(train_set, True)

        shared_kwargs = dict(drop_last=drop_last, num_workers=num_workers, pin_memory=pin_memory,
                             collate_fn=self.collate_fn)
        train_loader = DataLoader(train_set, batch_size, sampler=samplers[0], **shared_kwargs)
        test_loader = DataLoader(test_set, test_batch_size, sampler=samplers[2], **shared_kwargs)
        if use_prefetcher:
            train_loader = DataPrefetchWrapper(train_loader, start_epoch)
            test_loader = DataPrefetchWrapper(test_loader, start_epoch)

        ret = [train_loader, test_loader]

        if val_set is not None:
            if is_distributed():
                samplers[1] = DistributedSampler(val_set, **dist_sampler_kwargs)
                samplers[1].set_epoch(start_epoch)
            val_loader = DataLoader(val_set, test_batch_size, sampler=samplers[1], **shared_kwargs)
            if use_prefetcher:
                val_loader = DataPrefetchWrapper(test_loader)
            ret.append(val_loader)

        if return_num_classes:
            ret.append(self.num_classes)

        return tuple(ret)

    __call__ = get_dataloader

    @staticmethod
    def _split_dataset(train_set: datasets.VisionDataset,
                       val_size: int
                       ) -> (datasets.VisionDataset, datasets.VisionDataset):
        # split train_set to train_set and val_set
        assert len(train_set) >= val_size
        indices = torch.randperm(len(train_set))
        valset = copy.deepcopy(train_set)

        if hasattr(train_set, 'data'):
            train_set.data = [train_set.data[i] for i in indices[val_size:]]
        if hasattr(train_set, 'samples'):
            train_set.samples = [train_set.samples[i] for i in indices[val_size:]]

        train_set.targets = [train_set.targets[i] for i in indices[val_size:]]

        valset.data = [valset.data[i] for i in indices[:val_size]]
        valset.targets = [valset.targets[i] for i in indices[:val_size]]

        return train_set, valset

    @staticmethod
    def _sample_dataset(dataset: datasets.VisionDataset,
                        size: int
                        ) -> datasets.VisionDataset:
        indices = torch.randperm(len(dataset))[:size]
        if hasattr(dataset, 'data'):
            dataset.data = [dataset.data[i] for i in indices]
        if hasattr(dataset, 'samples'):
            # e.g., imagenet
            dataset.samples = [dataset.samples[i] for i in indices]
        dataset.targets = [dataset.targets[i] for i in indices]
        return dataset


DATASET_REGISTRY = Registry('vision_datasets', type=VisionSet)

DATASET_REGISTRY.register_from_dict(
    {  # fast_cifar10 uses fast_collate
        'fast_cifar10': VisionSet(datasets.CIFAR10, "~/.torch/data/cifar10", 10,
                                  [],
                                  [transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                                   transforms.RandomHorizontalFlip()],
                                  collate_fn=partial(fast_collate,
                                                     memory_format=torch.contiguous_format,
                                                     mean=(0.4914, 0.4822, 0.4465),
                                                     std=(0.2023, 0.1994, 0.2010))),

        'cifar10': VisionSet(datasets.CIFAR10, "~/.torch/data/cifar10", 10,
                             [transforms.ToTensor(),
                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))],
                             [transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                              transforms.RandomHorizontalFlip()]),

        'cifar100': VisionSet(datasets.CIFAR100, "~/.torch/data/cifar100", 100,
                              [transforms.ToTensor(),
                               transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))],
                              [transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                               transforms.RandomHorizontalFlip()]),

        'SVHN': VisionSet(OriginalSVHN, "~/.torch/data/svhn", 10,
                          [transforms.ToTensor(),
                           transforms.Normalize((0.4390, 0.4443, 0.4692), (0.1189, 0.1222, 0.1049))],
                          [transforms.RandomCrop(32, padding=4, padding_mode='reflect')]
                          ),

        'fast_imagenet': VisionSet(ImageNet, get_environ('IMAGENET_ROOT', '~/.torch/data/imagenet'), 1_000,
                                   [],
                                   [transforms.RandomResizedCrop(
                                       224), transforms.RandomHorizontalFlip()],
                                   [transforms.Resize(256), transforms.CenterCrop(224)],
                                   collate_fn=partial(fast_collate,
                                                      memory_format=torch.contiguous_format,
                                                      mean=(0.485, 0.456, 0.406),
                                                      std=(0.229, 0.224, 0.225))),

        'imagenet': VisionSet(ImageNet, get_environ('IMAGENET_ROOT', '~/.torch/data/imagenet'), 1_000,
                              [transforms.ToTensor(),
                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))],
                              [transforms.RandomResizedCrop(
                                  224), transforms.RandomHorizontalFlip()],
                              [transforms.Resize(256), transforms.CenterCrop(224)]),

    }
)
