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


def accuracy(input: torch.Tensor, target: torch.Tensor, reduction="mean") -> torch.Tensor:
    return _reduction((input.argmax(dim=-1) == target).float(),
                      reduction=reduction)


def true_positive(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = input.argmax(dim=-1)
    out = torch.zeros(input.size(-1))
    # todo: remove for loop
    for i in range(input.size(-1)):
        out[i] = ((pred == i) & (target == i)).sum()
    return out


def true_negative(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = input.argmax(dim=-1)
    out = torch.zeros(input.size(-1))
    for i in range(input.size(-1)):
        out[i] = ((pred != i) & (target != i)).sum()
    return out


def false_positive(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = input.argmax(dim=-1)
    out = torch.zeros(input.size(-1))
    for i in range(input.size(-1)):
        out[i] = ((pred == i) & (target != i)).sum()
    return out


def false_negative(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = input.argmax(dim=-1)
    out = torch.zeros(input.size(-1))
    for i in range(input.size(-1)):
        out[i] = ((pred != i) & (target == i)).sum()
    return out


def precision(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    tp = true_positive(input, target)
    fp = false_positive(input, target)
    if any(tp == 0) or any(fp == 0):
        logger.warning("Zero division")
    return tp / (tp + fp)


def recall(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    tp = true_positive(input, target)
    fn = false_negative(input, target)
    return tp / (tp + fn)


def specificity(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    tn = true_negative(input, target)
    fp = false_positive(input, target)
    return tn / (tn + fp)


def f1_score(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prec = precision(input, target)
    rec = recall(input, target)
    return 2 * prec * rec / (prec + rec)
