from __future__ import annotations

import copy
import inspect
import pathlib
from dataclasses import dataclass
from typing import Callable, List, Optional, Protocol, Tuple, Type

import torch
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from torchvision import transforms as VT

from homura import get_global_rank, get_world_size, is_distributed
from ..transforms import ConcatTransform, TransformBase


class VisionSetProtocol(Protocol):
    def __init__(self, root, train, transform, download): ...

    def __len__(self): ...


@dataclass
class VisionSet:
    """ Dataset abstraction for vision datasets. Use case ::

        data = DATASET_REGISTER("cifar10").setup(...)
        for img, label in data.tran_loader:
            ...

    """

    tv_class: Type[VisionSetProtocol]
    root: str or pathlib.Path
    num_classes: int
    default_norm: List
    default_train_da: Optional[List] = None
    default_test_da: Optional[List] = None
    collate_fn: Optional[Callable] = None

    def __post_init__(self):
        # _ is trainer
        args = {'root', 'train', 'transform', 'download'}
        _, *args_init = inspect.getfullargspec(self.tv_class.__init__).args
        _, *args_new = inspect.getfullargspec(self.tv_class.__new__).args
        if not (args <= set(args_init) or args <= set(args_new)):
            raise RuntimeError(f"tv_class is expected to have signiture of DataSet(root, train, transform, download),"
                               f"but {self.tv_class} has arguments of {args_init} instead.")
        self.root = pathlib.Path(self.root).expanduser()
        self.default_train_da = self.default_train_da or []
        self.default_test_da = self.default_test_da or []
        self._train_set = None
        self._train_loader = None
        self._val_set = None
        self._val_loader = None
        self._test_set = None
        self._test_loader = None

    @property
    def train_loader(self):
        return self._train_loader

    @property
    def val_loader(self):
        return self._val_loader

    @property
    def test_loader(self):
        return self._test_loader

    def setup(self,
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
              prefetch_factor: int = 2,
              persistent_workers: bool = False,
              worker_init_fn: Optional[Callable] = None,
              start_epoch: bool = 0
              ) -> VisionSet:
        vals = locals()
        vals.pop("self")
        self.get_dataloader(**vals)
        return self

    def get_dataset(self,
                    train_size: Optional[int] = None,
                    test_size: Optional[int] = None,
                    val_size: Optional[int] = None,
                    train_da: Optional[List] = None,
                    test_da: Optional[List] = None,
                    norm: Optional[List] = None,
                    download: bool = False,
                    pre_train_da: Optional[List] = None,
                    post_train_da: Optional[List] = None,
                    post_norm_train_da: Optional[List] = None
                    ) -> Tuple[VisionSetProtocol, VisionSetProtocol, Optional[VisionSetProtocol]]:
        """ Get Dataset

        :param train_size: Size of training dataset. If None, full dataset will be available.
        :param test_size: Size of test dataset. If None, full dataset will be available.
        :param val_size: Size of validation dataset, randomly split from the training dataset. If None, None will be returned.
        :param train_da: Data Augmentation for training
        :param test_da: Data Augmentation for testing and validation
        :param norm: Normalization after train_da and test_da
        :param download: If dataset needs downloading
        :param pre_train_da: Data Augmentation before the default data augmentation
        :param post_train_da: Data Augmentation after the default data augmentation
        :param post_norm_train_da: Data Augmentation after normalization (i.e., norm)
        :return: train_set, test_set, Optional[val_set]
        """

        assert (download or self.root.exists()), "root does not exist"
        train_da = train_da or list(self.default_train_da)
        test_da = test_da or list(self.default_test_da)
        norm = norm or list(self.default_norm)

        pre_train_da = pre_train_da or []
        post_train_da = post_train_da or []
        post_norm_train_da = post_norm_train_da or []

        train_transform = pre_train_da + train_da + post_train_da + norm + post_norm_train_da
        if any([isinstance(t, TransformBase) for t in train_transform]):
            train_transform = ConcatTransform(*train_transform)
        else:
            train_transform = VT.Compose(train_transform)
        train_set = self.tv_class(self.root, train=True, transform=train_transform, download=download)
        if train_size is not None and train_size > len(train_set):
            raise ValueError(f'train_size should be <={len(train_set)}')

        test_transform = test_da + norm
        if any(isinstance(t, TransformBase) for t in test_transform):
            test_transform = ConcatTransform(*test_transform)
        else:
            test_transform = VT.Compose(test_transform)
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
            if train_size > len(train_set):
                raise ValueError(f"train_size should be <= {len(train_set)}")
            train_set = self._sample_dataset(train_set, train_size)

        if test_size is not None:
            test_set = self._sample_dataset(test_set, test_size)

        self._train_set = train_set
        self._val_set = val_set
        self._test_set = test_set

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
                       prefetch_factor: int = 2,
                       persistent_workers: bool = False,
                       worker_init_fn: Optional[Callable] = None,
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
        :param prefetch_factor:
        :param persistent_workers:
        :param worker_init_fn:
        :param start_epoch: Epoch at start time
        :return: train_loader, test_loader, [val_loader], [num_classes]
        """

        train_set, test_set, val_set = self.get_dataset(train_size, test_size, val_size,
                                                        train_da, test_da, norm, download,
                                                        pre_train_da=pre_default_train_da,
                                                        post_train_da=post_default_train_da,
                                                        post_norm_train_da=post_norm_train_da)
        if test_batch_size is None:
            test_batch_size = non_training_bs_factor * batch_size

        samplers = [None, None, None]
        if is_distributed():

            dist_sampler_kwargs = dict(num_replicas=get_world_size(), rank=get_global_rank())
            samplers[0] = DistributedSampler(train_set, **dist_sampler_kwargs)
            samplers[2] = DistributedSampler(test_set, **dist_sampler_kwargs)
            samplers[0].set_epoch(start_epoch)
            samplers[2].set_epoch(start_epoch)
        else:
            samplers[0] = RandomSampler(train_set, True)

        shared_kwargs = dict(drop_last=drop_last, num_workers=num_workers, pin_memory=pin_memory,
                             collate_fn=self.collate_fn, prefetch_factor=prefetch_factor,
                             persistent_workers=persistent_workers, worker_init_fn=worker_init_fn)
        train_loader = DataLoader(train_set, batch_size, sampler=samplers[0], **shared_kwargs)
        test_loader = DataLoader(test_set, test_batch_size, sampler=samplers[2], **shared_kwargs)
        self._train_loader = train_loader
        self._test_loader = test_loader

        ret = [train_loader, test_loader]

        if val_set is not None:
            if is_distributed():
                samplers[1] = DistributedSampler(val_set, **dist_sampler_kwargs)
                samplers[1].set_epoch(start_epoch)
            val_loader = DataLoader(val_set, test_batch_size, sampler=samplers[1], **shared_kwargs)
            ret.append(val_loader)
            self._val_loader = val_loader

        if return_num_classes:
            ret.append(self.num_classes)

        return tuple(ret)

    __call__ = get_dataloader

    @staticmethod
    def _split_dataset(train_set: VisionSetProtocol,
                       val_size: int
                       ) -> (VisionSetProtocol, VisionSetProtocol):
        # split train_set to train_set and val_set
        assert len(train_set) >= val_size
        indices = torch.randperm(len(train_set))
        val_set = copy.deepcopy(train_set)

        if hasattr(train_set, 'data'):
            train_set.data = [train_set.data[i] for i in indices[val_size:]]
            val_set.data = [val_set.data[i] for i in indices[:val_size]]
        if hasattr(train_set, 'samples'):
            train_set.samples = [train_set.samples[i] for i in indices[val_size:]]
            val_set.samples = [val_set.samples[i] for i in indices[:val_size]]
            train_set.data = train_set.samples
            val_set.data = val_set.samples

        train_set.targets = [train_set.targets[i] for i in indices[val_size:]]
        val_set.targets = [val_set.targets[i] for i in indices[:val_size]]

        return train_set, val_set

    @staticmethod
    def _sample_dataset(dataset: VisionSetProtocol,
                        size: int
                        ) -> VisionSetProtocol:
        indices = torch.randperm(len(dataset))[:size]
        if hasattr(dataset, 'data'):
            dataset.data = [dataset.data[i] for i in indices]
        if hasattr(dataset, 'samples'):
            # e.g., imagenet
            dataset.samples = [dataset.samples[i] for i in indices]
        dataset.targets = [dataset.targets[i] for i in indices]
        return dataset
