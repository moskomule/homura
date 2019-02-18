# some functions to discretize input tensors
import random

import torch
from torch.autograd import Function
from torch.distributions import RelaxedOneHotCategorical, RelaxedBernoulli
from torch.nn import functional as F

__all__ = ["gumbel_softmax", "gumbel_sigmoid", "straight_through_estimator", "semantic_hashing"]


def gumbel_sigmoid(input: torch.Tensor, temp: float):
    return RelaxedBernoulli(temp, probs=input.sigmoid()).rsample()


class _STE(Function):
    """ Straight Through Estimator
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return F.hardtanh(grad_output)


def straight_through_estimator(input: torch.Tensor):
    return _STE.apply(input)


def _saturated_sigmoid(input: torch.Tensor):
    # max(0, min(1, 1.2 * input.sigmoid() - 0.1))
    return F.relu(1 - F.relu(1.1 - 1.2 * input.sigmoid()))


def semantic_hashing(input: torch.Tensor, is_training: bool):
    v1 = _saturated_sigmoid(input)
    v2 = (input < 0).float().detach()
    if not is_training or random.random() > 0.5:
        # derivative is 0 + dv1/dx + 0
        return v1 - v1.detach() + v2
    else:
        return v1


def gumbel_softmax(input: torch.Tensor, dim: int, temp: float) -> torch.Tensor:
    return RelaxedOneHotCategorical(temp, input.softmax(dim=dim)).rsample()
