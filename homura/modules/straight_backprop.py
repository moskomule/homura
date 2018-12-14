from torch.autograd import Function


def straight_backprop(function):
    """
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
