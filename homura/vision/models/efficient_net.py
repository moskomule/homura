from typing import Callable, Optional, Type

import torch
from torch import nn

from ._utils import conv1x1, conv3x3


class EffHead(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_classes: int,
                 dropout_ratio: float,
                 norm: Optional[Type[nn.BatchNorm2d]],
                 act: Callable[[torch.Tensor], torch.Tensor]
                 ):
        super().__init__()
        self.conv = conv1x1(in_channels, out_channels, bias=norm is None)
        self.norm = nn.Identity() if norm is None else norm(out_channels)
        self.act = act
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.fc = nn.Linear(out_channels, num_classes)

    def forward(self,
                input: torch.Tensor
                ) -> torch.Tensor:
        x = self.act(self.norm(self.conv(input)))
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(self.dropout(x))


class SEModule(nn.Module):
    def __init__(self,
                 in_channels: int,
                 se_channels: int,
                 act: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(conv1x1(in_channels, se_channels, bias=True),
                                act,
                                conv1x1(se_channels, in_channels, bias=True),
                                nn.Sigmoid())

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        return x * self.se(self.avg_pool(x))


class MBConv(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 exp_rate: float,
                 kernel_size: int,
                 stride: int,
                 se_rate: float,
                 norm: Optional[Type[nn.BatchNorm2d]],
                 act: Callable[[torch.Tensor], torch.Tensor]
                 ):
        super().__init__()
        self.exp = nn.Identity()
        self.act = act
        exp_channels = int(in_channels * exp_rate)
        if exp_channels != in_channels:
            self.exp = nn.Sequential(conv1x1(in_channels, exp_channels, bias=norm is None),
                                     nn.Identity() if norm is None else norm(exp_channels),
                                     act)
        self.depthwise = nn.Conv2d(exp_channels, exp_channels, kernel_size, stride=stride, groups=exp_channels,
                                   bias=norm is None)
        self.depthwise_norm = nn.Identity() if norm is None else norm(exp_channels)
        self.se = SEModule(exp_channels, int(in_channels * se_rate), self.act)
        self.lin_proj = conv1x1(exp_channels, out_channels, bias=norm is None)
        self.lin_proj_norm = nn.Identity if norm is None else norm(out_channels)
        self.has_skip = (stride == 1) and (in_channels == out_channels)

    def forward(self,
                input: torch.Tensor
                ) -> torch.Tensor:
        x = self.exp(input)
        x = self.act(self.depthwise_norm(self.depthwise(x)))
        x = self.se(x)
        x = self.lin_proj_norm(self.lin_proj(x))
        if self.has_skip:
            x = x + input
        return x


class EfficientNetStage(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 exp_rate: float,
                 kernel_size: int,
                 stride: int,
                 se_rate: float,
                 depth: int,
                 norm: Optional[Type[nn.BatchNorm2d]],
                 act: Callable[[torch.Tensor], torch.Tensor]
                 ):
        layers = []
        for i in range(depth):
            layers.append(MBConv(in_channels, out_channels, exp_rate, kernel_size, stride, se_rate, norm, act))
            stride, in_channels = 1, out_channels
        super().__init__(*layers)


class StemIN(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm: Optional[Type[nn.BatchNorm2d]],
                 act: Callable[[torch.Tensor], torch.Tensor]
                 ):
        super().__init__(conv3x3(in_channels, out_channels, stride=2, bias=norm is None),
                         nn.Identity() if norm is None else norm(out_channels),
                         act)


class EfficientNet(nn.Module):
    def __init__(self,
                 stem_channels: int,
                 norm: Optional[Type[nn.BatchNorm2d]],
                 act: Callable[[torch.Tensor], torch.Tensor]
                 ):
        super().__init__()

        self.stem = StemIN(3, stem_channels, norm, act)

