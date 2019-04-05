from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F


def weight_standardization(model: nn.Module,
                           input: torch.Tensor) -> torch.Tensor:
    """ `forward` of weight initialization. Currently only supports `nn.Conv2d`.

    :param model:
    :param input:
    :return:
    """

    if not isinstance(model, nn.Conv2d):
        raise RuntimeError(f"`model` is expected to be `nn.Conv2d`, but got {type(model)}")
    weight = model.weight
    # should be faster than original implementation
    mean = weight.view(weight.size(0), -1, 1, 1).mean(dim=1, keepdim=True)
    std = weight.view(weight.size(0), -1, 1, 1).std(dim=1, keepdim=True) + 1e-5
    weight = (weight - mean) / std.expand_as(weight)
    return F.conv2d(input, weight, model.bias, model.stride,
                    model.padding, model.dilation, model.groups)


def convert_ws(model: nn.Module) -> nn.Module:
    """ Convert the given model's `nn.Conv2d` to weight standardized `Conv2d`

    :param model:
    :return:
    """

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            setattr(module, "forward", MethodType(weight_standardization, module))
    return model


class WSConv2d(nn.Conv2d):
    """ `nn.Conv2d` with weight standardization

    """

    def forward(self,
                input: torch.Tensor) -> torch.Tensor:
        return weight_standardization(self, input)
