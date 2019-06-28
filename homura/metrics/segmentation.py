# some metrics especially for semantic/instance segmentation

import torch

from homura.modules import to_onehot
from .commons import confusion_matrix

__all__ = ["binary_as_multiclass", "pixel_accuracy", "mean_iou", "classwise_iou"]


def binary_as_multiclass(input: torch.Tensor,
                         threshold: float) -> torch.Tensor:
    """ Convert `Bx1xHxW` outputs to `BxCxHxW`.

    :param input:
    :param threshold:
    :return:
    """
    if input.size(1) != 1:
        raise RuntimeError(f"Channel dimension is expected to be 1, but got {input.size(1)}")
    return torch.cat([input.clone().fill_(threshold), input], dim=1)


def pixel_accuracy(input: torch.Tensor,
                   target: torch.Tensor) -> torch.Tensor:
    """ Pixel accuracy

    :param input: logits (`BxCxHxW`)
    :param target: target in LongTensor (`BxHxW`)
    :return:
    """
    if input.dim() != 4:
        raise RuntimeError(f"Dimension of input is expected to be 4, but got {input.dim()}")
    if target.dim() != 3:
        raise RuntimeError(f"Dimension of target is expected to be 3, but got {target.dim()}")

    b, c, h, w = input.size()
    pred = to_onehot(input.argmax(dim=1), num_classes=c)
    gt = to_onehot(target, num_classes=c)
    acc = (pred * gt).sum(dim=(1, 2, 3)) / (w * h)
    return acc.mean()


def classwise_iou(input: torch.Tensor,
                  target: torch.Tensor) -> torch.Tensor:
    """ Class-wise IoU

    :param input: logits (`BxCxHxW`)
    :param target: target in LongTensor (`BxHxW`)
    :return:
    """
    if input.dim() != 4:
        raise RuntimeError(f"Dimension of input is expected to be 4, but got {input.dim()}")
    if target.dim() != 3:
        raise RuntimeError(f"Dimension of target is expected to be 3, but got {target.dim()}")

    cm = confusion_matrix(input, target).float()
    print(cm)
    return cm.diag() / (cm.sum(0) + cm.sum(1) - cm.diag())


def mean_iou(input: torch.Tensor,
             target: torch.Tensor) -> torch.Tensor:
    """ Mean IoU

    :param input: logits (`BxCxHxW`)
    :param target: target in LongTensor (`BxHxW`)
    :return:
    """
    return classwise_iou(input, target).mean()
