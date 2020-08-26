# some functions to discretize input_forward tensors

import random

import torch
from torch.autograd import Function
from torch.distributions import RelaxedBernoulli
from torch.nn import functional as F

__all__ = ["gumbel_sigmoid", "straight_through_estimator", "semantic_hashing"]


def gumbel_sigmoid(input: torch.Tensor,
                   temp: float) -> torch.Tensor:
    """ gumbel sigmoid function
    """
    return RelaxedBernoulli(temp, probs=input.sigmoid()).rsample()


class _STE(Function):
    """ Straight Through Estimator
    """

    @staticmethod
    def forward(ctx,
                input: torch.Tensor) -> torch.Tensor:
        return (input > 0).float()

    @staticmethod
    def backward(ctx,
                 grad_output: torch.Tensor) -> torch.Tensor:
        return F.hardtanh(grad_output)


def straight_through_estimator(input: torch.Tensor) -> torch.Tensor:
    """ straight through estimator

    >>> straight_through_estimator(torch.randn(3, 3))
    tensor([[0., 1., 0.],
            [0., 1., 1.],
            [0., 0., 1.]])
    """
    return _STE.apply(input)


def _saturated_sigmoid(input: torch.Tensor) -> torch.Tensor:
    # max(0, min(1, 1.2 * input_forward.sigmoid() - 0.1))
    return F.relu(1 - F.relu(1.1 - 1.2 * input.sigmoid()))


def semantic_hashing(input: torch.Tensor, is_training: bool) -> torch.Tensor:
    """ Semantic hashing

    >>> semantic_hashing(torch.randn(3, 3), True) # by 0.5
    tensor([[0.3515, 0.0918, 0.7717],
            [0.8246, 0.1620, 0.0689],
            [1.0000, 0.3575, 0.6598]])

    >>> semantic_hashing(torch.randn(3, 3), False)
    tensor([[0., 0., 1.],
            [0., 1., 1.],
            [0., 1., 1.]])
    """
    v1 = _saturated_sigmoid(input)
    v2 = (input < 0).float().detach()
    if not is_training or random.random() > 0.5:
        # derivative is 0 + dv1/dx + 0
        return v1 - v1.detach() + v2
    else:
        return v1
