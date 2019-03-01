# some metrics especially for semantic/instance segmentation

import torch

from homura.modules import to_onehot

__all__ = ["binary_to_multiclass", "pixel_accuracy", "mean_iou", "classwise_iou"]


def binary_to_multiclass(input: torch.Tensor, threshold: float):
    """ Convert `BxHxW' outputs to `BxCxHxW`

    :param input:
    :param threshold:
    :return:
    """
    return torch.stack([input.clone().fill_(threshold),
                        input], dim=1)


def pixel_accuracy(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """

    :param input: logits (`BxCxHxW`)
    :param target: target in LongTensor (`BxHxW`)
    :return:
    """
    b, c, h, w = input.size()
    pred = to_onehot(input.argmax(dim=1), num_classes=c)
    gt = to_onehot(target, num_classes=c)
    acc = (pred * gt).view(b, -1).sum(-1) / (w * h)
    return acc.mean()


def classwise_iou(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """

    :param input:
    :param target:
    :return:
    """
    b, c, h, w = input.size()
    pred = to_onehot(input.argmax(dim=1), num_classes=c)
    gt = to_onehot(target, num_classes=c)
    tp = (pred * gt).view(b, c, -1).sum(-1)
    ps = pred.view(b, c, -1).sum(-1)
    tr = gt.view(b, c, -1).sum(-1)
    return (tp / (ps + tr - tp + 1e-8)).mean(0)


def mean_iou(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return classwise_iou(input, target).mean()
