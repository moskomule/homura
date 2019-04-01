import torch
from torch import nn
from torch.autograd import Function

__all__ = ["straight_backprop"]


def straight_backprop(function):
    """ A function whose `derivative` is as linear

    >>> straight_backprop_relu = straight_backprop(F.relu)
    >>> straight_backprop_relu(tensor)

    :param function: original function
    :return: modified function
    """

    class _StraightBackprop(Function):
        @staticmethod
        def forward(ctx, inputs):
            return function(inputs)

        @staticmethod
        def backward(ctx, grad_outputs):
            return grad_outputs

    return _StraightBackprop.apply


class StraightBackprop(nn.Module):
    """ A function whose `derivative` is as linear

    """

    def __init__(self,
                 function):
        super(StraightBackprop, self).__init__()
        self._fn = straight_backprop(function)

    def forward(self, input) -> torch.Tensor:
        return self._fn(input)
