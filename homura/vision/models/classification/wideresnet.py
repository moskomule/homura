# WideResNet proposed in http://arxiv.org/abs/1605.07146

import torch.nn as nn
import torch.nn.functional as F

from . import MODEL_REGISTRY
from .._intialization import init_parameters

__all__ = ["WideResNet", "WideBasicModule", "wrn28_10", "wrn28_2"]


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


class WideBasicModule(nn.Module):

    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(WideBasicModule, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, (3, 3), 1, 1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, (3, 3), stride, 1)

        self.shortcut = None
        if stride != 1 or in_planes != planes:
            self.shortcut = conv1x1(in_planes, planes, stride)

    def forward(self, x):
        residual = x
        x = self.dropout(self.conv1(F.relu(self.bn1(x))))
        x = self.conv2(F.relu(self.bn2(x)))
        if self.shortcut is None:
            x += residual
        else:
            x += self.shortcut(residual)
        return x


class WideResNet(nn.Module):
    """WideResNet for CIFAR data.
    """

    def __init__(self, num_classes, depth, widen_factor, dropout_rate, base=16):
        super(WideResNet, self).__init__()
        self.in_planes = base

        assert ((depth - 4) % 6 == 0), "depth should be 6n+4"
        n = (depth - 4) // 6
        k = widen_factor

        num_stages = [base, base * k, base * k * 2, base * k * 4]

        self.conv1 = conv3x3(3, num_stages[0])
        self.layer1 = self._wide_layer(WideBasicModule, num_stages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideBasicModule, num_stages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(WideBasicModule, num_stages[3], n, dropout_rate, stride=2)
        self.bn = nn.BatchNorm2d(num_stages[3])
        self.fc = nn.Linear(num_stages[3], num_classes)
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

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * int(num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.relu(self.bn(x))
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
