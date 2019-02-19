import torch
from torch import nn
from torch.nn import functional as F

from .._intialization import init_parameters

__all__ = ["msdnet25", "msdnet50"]


class MSDBase(nn.Module):
    def __init__(self, in_channels, dilation):
        super(MSDBase, self).__init__()
        self.dilation = dilation
        self.conv = nn.Conv2d(
            in_channels, out_channels=1, kernel_size=3, dilation=self.dilation)

    def forward(self, input):
        pad = (self.dilation,) * 4
        x = F.pad(input, pad, mode="reflect")
        return F.relu(self.conv(x))


class MSDLayer(nn.Module):
    def __init__(self, input_channels, i, width):
        super(MSDLayer, self).__init__()
        _in_channels = i * width + input_channels
        self.layers = nn.ModuleList([
            MSDBase(_in_channels, (i * width + j) % 10 + 1)
            for j in range(1, width + 1)
        ])

    def forward(self, input):
        outs = torch.cat([m(input) for m in self.layers], dim=1)
        return outs


class MSDNet(nn.Module):
    def __init__(self, input_channels, out_channels, depth, width):
        super(MSDNet, self).__init__()
        self.layers = nn.ModuleList(
            [MSDLayer(input_channels, i, width) for i in range(depth)])
        self.conv = nn.Conv2d(
            width * depth + input_channels, out_channels, kernel_size=1)
        init_parameters(self)

    def forward(self, input):
        for m in self.layers:
            out = m(input)
            input = torch.cat([input, out], dim=1)
        return self.conv(input)


def msdnet25(num_classes, input_channels=3):
    return MSDNet(input_channels, num_classes, 25, 1)


def msdnet50(num_classes, input_channels=3):
    return MSDNet(input_channels, num_classes, 50, 1)
