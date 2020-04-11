from typing import Tuple, List, Optional

from torch.utils.data import DataLoader, RandomSampler, DistributedSampler

from .datasets import _split_dataset, _DATASETS


def vision_loaders(name: str,
                   batch_size: int,
                   train_da: Optional[List] = None,
                   test_da: Optional[List] = None,
                   norm: Optional[List] = None,
                   val_size: int = 0,
                   download: bool = False,
                   num_workers: int = -1,
                   non_training_bs_factor=2,
                   distributed: bool = False,
                   return_num_classes: bool = False) -> Tuple:
    """ Get data loaders for registered vision datasets. homura expects datasets are in `~/.torch/data/DATASET_NAME`.
    Link path if necessary, e.g. `ln -s /original/path $HOME/.torch`. Datasets can be registered
    using `homura.vision.register_dataset`

    :param name: name of dataset.
    :param batch_size:
    :param train_da: custom train-time data augmentation
    :param test_da: custom test-time data augmentation
    :param norm: custom normalization after train_da/test_da
    :param val_size: If `val_size>0`, split train set
    :param download:
    :param num_workers:
    :param non_training_bs_factor:
    :param distributed:
    :param return_num_classes:
    :return: (train_set, test_set, [val_set], [num_classes])
    """

    if name not in _DATASETS.keys():
        raise RuntimeError(f'Unknown dataset name {name}.')
    dataset = _DATASETS[name]
    train_set, test_set = dataset.customize(train_da, test_da, norm, download)
    if val_size > 0:
        train_set, val_set = _split_dataset(train_set, val_size)
        val_set.transform = test_set.transform

    samplers = [None, None, None]
    if distributed:
        import homura

        kwargs = dict(num_replicas=homura.get_world_size(), rank=homura.get_global_rank())
        samplers[0] = DistributedSampler(train_set, **kwargs)
        samplers[2] = DistributedSampler(test_set, **kwargs)
    else:
        samplers[0] = RandomSampler(train_set, True)

    train_loader = DataLoader(train_set, batch_size,
                              sampler=samplers[0],
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, non_training_bs_factor * batch_size, sampler=samplers[2],
                             num_workers=num_workers, pin_memory=True)
    ret = [train_loader, test_loader]
    if val_size > 0:
        if distributed:
            samplers[1] = DistributedSampler(test_set, **kwargs)
        val_loader = DataLoader(val_set, non_training_bs_factor * batch_size, sampler=samplers[1],
                                num_workers=num_workers, pin_memory=True)
        ret.append(val_loader)

    if return_num_classes:
        ret.append(dataset.num_classes)

    return tuple(ret)
