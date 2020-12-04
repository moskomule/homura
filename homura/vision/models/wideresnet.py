# WideResNet proposed in http://arxiv.org/abs/1605.07146
from typing import Callable, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from homura.vision.models import MODEL_REGISTRY
from ._utils import conv1x1, conv3x3, init_parameters

__all__ = ["WideResNet", "WideBasicModule", "wrn28_10", "wrn28_2"]


class WideBasicModule(nn.Module):
    def __init__(self,
                 in_planes: int,
                 planes: int,
                 dropout_rate: float,
                 stride: int,
                 norm: Optional[Type[nn.BatchNorm2d]],
                 act: Callable[[torch.Tensor], torch.Tensor]
                 ):
        super().__init__()
        self.act = act
        if norm is None:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        else:
            self.norm1 = norm(num_features=in_planes)
            self.norm2 = norm(num_features=planes)
        self.conv1 = conv3x3(in_planes, planes, bias=norm is None)
        self.conv2 = conv3x3(planes, planes, stride=stride, bias=norm is None)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = conv1x1(in_planes, planes, stride, bias=norm is None)

    def forward(self,
                input: torch.Tensor
                ) -> torch.Tensor:
        out = self.conv1(self.act(self.norm1(input)))
        out = self.dropout(out)
        out = self.conv2(self.act(self.norm2(out)))
        out += self.shortcut(input)
        return out


class WideResNet(nn.Module):
    def __init__(self,
                 num_classes: int,
                 depth: int,
                 widen_factor: int,
                 in_channels: int = 3,
                 norm: Optional[Type[nn.BatchNorm2d]] = nn.BatchNorm2d,
                 act: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU(),
                 base: int = 16,
                 dropout_rate: float = 0
                 ):
        super().__init__()
        assert (depth - 4) % 6 == 0
        self.in_planes = base
        self.norm = norm
        self.act = act
        self.depth = depth

        n = (depth - 4) // 6
        k = widen_factor

        num_stages = [base, base * k, base * k * 2, base * k * 4]

        self.conv1 = conv3x3(in_channels, num_stages[0])
        self.layer1 = self._wide_layer(WideBasicModule, num_stages[1], n, dropout_rate, stride=1, norm=norm)
        self.layer2 = self._wide_layer(WideBasicModule, num_stages[2], n, dropout_rate, stride=2, norm=norm)
        self.layer3 = self._wide_layer(WideBasicModule, num_stages[3], n, dropout_rate, stride=2, norm=norm)
        self.last_dim = num_stages[3]
        if norm is None:
            self.norm = nn.Identity()
        else:
            self.norm = norm(num_features=self.last_dim)
        self.fc = nn.Linear(self.last_dim, num_classes)
        self.initialize()

    def initialize(self):
        # following author's implementation
        init_parameters(self)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(getattr(m, "running_mean"), 0)
                nn.init.constant_(getattr(m, "running_var"), 1)

    def _wide_layer(self,
                    block: Type[WideBasicModule],
                    planes: int,
                    num_blocks: int,
                    dropout_rate: float,
                    stride: int,
                    norm: Optional[Type[nn.BatchNorm2d]],
                    ) -> nn.Sequential:
        strides = [stride] + [1] * int(num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride, norm, self.act))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.act(self.norm(x))
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


@MODEL_REGISTRY.register
def wrn28_10(num_classes=10, dropout_rate=0) -> WideResNet:
    model = WideResNet(depth=28, widen_factor=10, dropout_rate=dropout_rate, num_classes=num_classes)
    return model


@MODEL_REGISTRY.register
def wrn28_2(num_classes=10, dropout_rate=0) -> WideResNet:
    model = WideResNet(depth=28, widen_factor=2, dropout_rate=dropout_rate, num_classes=num_classes)
    return model


@MODEL_REGISTRY.register
def wrn40_2(num_classes=10, dropout_rate=0) -> WideResNet:
    model = WideResNet(depth=40, widen_factor=2, dropout_rate=dropout_rate, num_classes=num_classes)
    return model
