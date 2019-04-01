import torch
from torch import nn

from homura.modules.regularizer import convert_ws


def test_convert_ws():
    input = torch.randn(1, 3, 32, 32)
    conv2d = nn.Conv2d(3, 16, 3)
    wsconv = convert_ws(conv2d)
    output = wsconv(input)
    output.sum().backward()
