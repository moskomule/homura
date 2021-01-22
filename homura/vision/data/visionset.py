import copy
import inspect
import pathlib
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Tuple, Type

import torch
from torch.jit.annotations import List
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from torchvision import transforms as VT

from homura import is_distributed
from .prefetcher import DataPrefetchWrapper
from ..transforms import ConcatTransform, TransformBase


class VisionSetProtocol(Protocol):
    def __init__(self, root, train, transform, download): ...

    def __len__(self): ...


@dataclass
class VisionSet:
    """ Dataset abstraction for vision datasets.
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
                    ) -> Tuple[VisionSetProtocol, VisionSetProtocol, Optional[VisionSetProtocol]]:
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

        train_transform = pre_default_train_da + train_da + post_default_train_da + norm + post_norm_train_da
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
    def _split_dataset(train_set: VisionSetProtocol,
                       val_size: int
                       ) -> (VisionSetProtocol, VisionSetProtocol):
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
