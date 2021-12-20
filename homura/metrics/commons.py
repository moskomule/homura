from __future__ import annotations

import torch
from torch import Tensor

from homura.liblog import get_logger

logger = get_logger(__name__)

__all__ = ["true_positive", "true_negative", "false_positive", "false_negative",
           "classwise_accuracy", "precision", "recall", "specificity", "f1_score",
           "confusion_matrix", "accuracy"]


def _base(input: Tensor,
          target: Tensor
          ) -> tuple[Tensor, Tensor, Tensor]:
    classes = torch.arange(input.size(1), device=input.device)
    pred = input.argmax(dim=1).view(-1, 1)
    target = target.view(-1, 1)
    return pred, target, classes


def true_positive(input: Tensor,
                  target: Tensor
                  ) -> Tensor:
    """Calculate true positive

    :param input: output of network, expected to be `BxCx(OPTIONAL DIMENSIONS)`
    :param target: target, expected to be `Bx(OPTIONAL DIMENSIONS)`
    :return: true positive in float tensor of `C`
    """

    pred, target, classes = _base(input, target)
    out = (pred == classes) & (target == classes)
    return out.sum(dim=0).float()


def true_negative(input: Tensor,
                  target: Tensor
                  ) -> Tensor:
    """Calculate true negative

    :param input: output of network, expected to be `BxCx(OPTIONAL DIMENSIONS)`
    :param target: target, expected to be `Bx(OPTIONAL DIMENSIONS)`
    :return: true negative in float tensor of `C`
    """

    pred, target, classes = _base(input, target)
    out = ((pred != classes) & (target != classes))
    return out.sum(dim=0).float()


def false_positive(input: Tensor,
                   target: Tensor
                   ) -> Tensor:
    """Calculate false positive

    :param input: output of network, expected to be `BxCx(OPTIONAL DIMENSIONS)`
    :param target: target, expected to be `Bx(OPTIONAL DIMENSIONS)`
    :return: false positive in float tensor of `C`
    """

    pred, target, classes = _base(input, target)
    out = ((pred == classes) & (target != classes))
    return out.sum(dim=0).float()


def false_negative(input: Tensor,
                   target: Tensor
                   ) -> Tensor:
    """Calculate false negative
    :param input: output of network, expected to be `BxCx(OPTIONAL DIMENSIONS)`
    :param target: target, expected to be `Bx(OPTIONAL DIMENSIONS)`
    :return: false negative in float tensor of `C`
    """

    pred, target, classes = _base(input, target)
    out = ((pred != classes) & (target == classes))
    return out.sum(dim=0).float()


def classwise_accuracy(input: Tensor,
                       target: Tensor
                       ) -> Tensor:
    """Calculate class wise accuracy

    :param input: output of network, expected to be `BxCx(OPTIONAL DIMENSIONS)`
    :param target: target, expected to be `Bx(OPTIONAL DIMENSIONS)`
    :return: class wise accuracy in float tensor of `C`
    """

    tp = true_positive(input, target)
    tn = true_negative(input, target)
    fp = false_positive(input, target)
    fn = false_negative(input, target)
    denom = tp + tn + fp + fn
    if any(denom == 0):
        logger.warning("Zero division in accuracy")
    return (tp + tn) / denom


def precision(input: Tensor,
              target: Tensor
              ) -> Tensor:
    """Calculate precision

    :param input: output of network, expected to be `BxCx(OPTIONAL DIMENSIONS)`
    :param target: target, expected to be `Bx(OPTIONAL DIMENSIONS)`
    :return: precision in float tensor of `C`
    """

    tp = true_positive(input, target)
    fp = false_positive(input, target)
    denom = tp + fp
    if any(denom == 0):
        logger.warning("Zero division in precision")
    return tp / denom


def recall(input: Tensor,
           target: Tensor
           ) -> Tensor:
    """Calculate recall

    :param input: output of network, expected to be `BxCx(OPTIONAL DIMENSIONS)`
    :param target: target, expected to be `Bx(OPTIONAL DIMENSIONS)`
    :return: recall in float tensor of `C`
    """

    tp = true_positive(input, target)
    fn = false_negative(input, target)
    denom = tp + fn
    if any(denom == 0):
        logger.warning("Zero division in recall")
    return tp / denom


def specificity(input: Tensor,
                target: Tensor
                ) -> Tensor:
    """Calculate specificity

    :param input: output of network, expected to be `BxCx(OPTIONAL DIMENSIONS)`
    :param target: target, expected to be `Bx(OPTIONAL DIMENSIONS)`
    :return: specificity in float tensor of `C`
    """

    tn = true_negative(input, target)
    fp = false_positive(input, target)
    denom = tn + fp
    if any(denom == 0):
        logger.warning("Zero division in specificity")
    return tn / denom


def f1_score(input: Tensor,
             target: Tensor
             ) -> Tensor:
    """Calculate f1 score

    :param input: output of network, expected to be `BxCx(OPTIONAL DIMENSIONS)`
    :param target: target, expected to be `Bx(OPTIONAL DIMENSIONS)`
    :return: f1 score in float tensor of `C`
    """

    prec = precision(input, target)
    rec = recall(input, target)
    return 2 * prec * rec / (prec + rec)


def confusion_matrix(input: Tensor,
                     target: Tensor
                     ) -> Tensor:
    """Calculate confusion matrix

    :param input: output of network, expected to be `BxCx(OPTIONAL DIMENSIONS)`
    :param target: target, expected to be `Bx(OPTIONAL DIMENSIONS)`
    :return: confusion matrix in long tensor of `CxC`
    """

    num_classes = input.size(1)
    indices = (0 <= target) & (target < num_classes)
    pred = input.argmax(dim=1)[indices]
    inds = num_classes * pred + target[indices]
    return inds.bincount(minlength=num_classes ** 2).view(num_classes, num_classes)


@torch.no_grad()
def accuracy(input: Tensor,
             target: Tensor,
             top_k: int = 1
             ) -> Tensor:
    if target.ndim == 2:
        target = target.argmax(dim=-1)
    pred_idx = input.argmax(dim=-1, keepdim=True) if top_k == 1 else input.topk(k=top_k, dim=-1).indices
    target = target.view(-1, 1).expand_as(pred_idx)
    return (pred_idx == target).float().sum(dim=1).mean()
