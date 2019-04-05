# functions to convert given tensors

import torch

__all__ = ["to_onehot"]


def to_onehot(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """ Convert given target tensor to onehot format

    :param target: `LongTensor` of `BxCx(optional dimensions)`
    :param num_classes: number of classes
    :return:
    """

    size = list(target.size())
    size.insert(1, num_classes)
    onehot = target.new_zeros(*size).float()
    return onehot.scatter_(1, target.unsqueeze(1), 1)
