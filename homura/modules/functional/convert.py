# functions to convert given tensors

import warnings

import torch
from torch.nn import functional as F

__all__ = ["to_onehot"]


def to_onehot(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """ Convert given target tensor to onehot format

    :param target: `LongTensor` of `BxCx(optional dimensions)`
    :param num_classes: number of classes
    :return:
    """
    warnings.warn("homura's to_onehot is now deprecated in favor of F.one_hot",
                  DeprecationWarning)
    return F.one_hot(target, num_classes)
