from typing import Tuple

import torch

from homura.liblog import get_logger

EPS = 1e-8

logger = get_logger(__name__)


def _reduction(input: torch.Tensor, reduction: str):
    if reduction == "mean":
        return input.mean()
    elif reduction == "sum":
        return input.sum()
    elif reduction == "none" or reduction is None:
        return input
    else:
        raise NotImplementedError(f"Wrong reduction: {reduction}")


def _base(input: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor]:
    classes = torch.arange(input.size(1), device=input.device)
    pred = input.argmax(dim=1).view(-1, 1)
    target = target.view(-1, 1)
    return pred, target, classes


def true_positive(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred, target, classes = _base(input, target)
    out = (pred == classes) & (target == classes)
    return out.sum(dim=0).float()


def true_negative(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred, target, classes = _base(input, target)
    out = ((pred != classes) & (target != classes))
    return out.sum(dim=0).float()


def false_positive(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred, target, classes = _base(input, target)
    out = ((pred == classes) & (target != classes))
    return out.sum(dim=0).float()


def false_negative(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred, target, classes = _base(input, target)
    out = ((pred != classes) & (target == classes))
    return out.sum(dim=0).float()


def classwise_accuracy(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    tp = true_positive(input, target)
    tn = true_negative(input, target)
    fp = false_positive(input, target)
    fn = false_negative(input, target)
    denom = tp + tn + fp + fn
    if any(denom == 0):
        logger.warning("Zero division in accuracy")
    return (tp + tn) / denom


def precision(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    tp = true_positive(input, target)
    fp = false_positive(input, target)
    denom = tp + fp
    if any(denom == 0):
        logger.warning("Zero division in precision")
    return tp / denom


def recall(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    tp = true_positive(input, target)
    fn = false_negative(input, target)
    denom = tp + fn
    if any(denom == 0):
        logger.warning("Zero division in recall")
    return tp / denom


def specificity(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    tn = true_negative(input, target)
    fp = false_positive(input, target)
    denom = tn + fp
    if any(denom == 0):
        logger.warning("Zero division in specificity")
    return tn / denom


def f1_score(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prec = precision(input, target)
    rec = recall(input, target)
    return 2 * prec * rec / (prec + rec)


import torch


def confusion_matrix(input: torch.Tensor, target: torch.Tensor):
    num_classes = input.size(1)
    classes = torch.arange(num_classes, device=input.device)
    pred = input.argmax(dim=1).view(-1, 1)
    target = target.view(-1, 1)
    return ((pred == classes).t() @ (target == classes)).long()
