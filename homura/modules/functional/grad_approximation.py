from __future__ import annotations

import torch
from torch.autograd import Function


class _CustomSTE(Function):
    """ An efficient alternatives for

    >>> input_forward.requires_grad is False
    >>> input_backward.requires_grad is True
    >>> input_forward + (input_backward - input_backward.detach())
    """

    @staticmethod
    def forward(ctx,
                input_forward: torch.Tensor,
                input_backward: torch.Tensor) -> torch.Tensor:
        ctx.shape = input_backward.shape
        return input_forward

    @staticmethod
    def backward(ctx,
                 grad_in: torch.Tensor) -> tuple[None, torch.Tensor]:
        return None, grad_in.sum_to_size(ctx.shape)


def custom_straight_through_estimator(input_forward: torch.Tensor,
                                      input_backward: torch.Tensor) -> torch.Tensor:
    return _CustomSTE.apply(input_forward, input_backward)
