"""
ResNet for CIFAR dataset proposed in He+15, p 7. and
https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua
"""
from typing import Callable, Optional, Type

import torch
from torch import nn

from homura.vision.models import MODEL_REGISTRY
from homura.vision.models._utils import conv1x1, conv3x3, init_parameters

__all__ = ["resnet20", "resnet32", "resnet56", "resnet110",
           "preact_resnet20", "preact_resnet32", "preact_resnet56", "preact_resnet110",
           "ResNet", "PreActResNet"]


def initialization(module: nn.Module,
                   use_zero_init: bool):
    init_parameters(module)
    if use_zero_init:
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in module.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.norm2.weight, 0)


class BasicBlock(nn.Module):
    def __init__(self,
                 in_planes: int,
                 planes: int,
                 stride: int,
                 norm: Optional[Type[nn.BatchNorm2d]],
                 act: Callable[[torch.Tensor], torch.Tensor]
                 ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride, bias=norm is None)
        self.conv2 = conv3x3(planes, planes, bias=norm is None)
        self.act = act
        self.stride = stride

        if norm is None:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        else:
            self.norm1 = norm(num_features=planes)
            self.norm2 = norm(num_features=planes)

        self.downsample = nn.Identity()
        if in_planes != planes:
            _norm = nn.Identity() if norm is None else norm(num_features=planes)
            self.downsample = nn.Sequential(conv1x1(in_planes, planes, bias=norm is None),
                                            _norm)

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out += residual
        out = self.act(out)

        return out


class PreActBasicBlock(BasicBlock):
    def __init__(self,
                 in_planes: int,
                 planes: int,
                 stride: int,
                 norm: Optional[Type[nn.BatchNorm2d]],
                 act: Callable[[torch.Tensor], torch.Tensor]
                 ):
        super(PreActBasicBlock, self).__init__(in_planes, planes, stride, norm, act)
        if norm is not None:
            self.norm1 = norm(num_features=in_planes)

    def forward(self, x):
        residual = self.downsample(x)
        out = self.norm1(x)
        out = self.act(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = self.act(out)
        out = self.conv2(out)

        out += residual

        return out


class ResNet(nn.Module):
    """ResNet for CIFAR data. For ImageNet classification, use `torchvision`'s.
    """

    def __init__(self,
                 num_classes: int,
                 depth: int,
                 in_channels: int = 3,
                 norm: Optional[Type[nn.BatchNorm2d]] = nn.BatchNorm2d,
                 act: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU(),
                 block: Type[BasicBlock] = None,
                 ):
        super(ResNet, self).__init__()
        assert (depth - 2) % 6 == 0
        n_size = (depth - 2) // 6
        self.inplane = 16
        self.conv1 = nn.Conv2d(in_channels, self.inplane, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.Identity() if norm is None else norm(self.inplane)
        self.norm = norm
        self.act = act
        self.layer1 = self._make_layer(block, 16, blocks=n_size, stride=1)
        self.layer2 = self._make_layer(block, 32, blocks=n_size, stride=2)
        self.layer3 = self._make_layer(block, 64, blocks=n_size, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
        initialization(self, False)

    def _make_layer(self, block: Type[BasicBlock], planes: int, blocks: int, stride: int):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplane, planes, stride, self.norm, self.act))
            self.inplane = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class PreActResNet(ResNet):
    def __init__(self, *args, **kwargs):
        super(PreActResNet, self).__init__(*args, **kwargs)
        self.norm1 = self.norm(self.inplane)
        initialization(self, False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.norm1(x)
        x = self.act(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


@MODEL_REGISTRY.register
def resnet20(**kwargs):
    model = ResNet(depth=20, block=BasicBlock, **kwargs)
    return model


@MODEL_REGISTRY.register
def resnet32(**kwargs):
    model = ResNet(depth=32, block=BasicBlock, **kwargs)
    return model


@MODEL_REGISTRY.register
def resnet56(**kwargs):
    model = ResNet(depth=56, block=BasicBlock, **kwargs)
    return model


@MODEL_REGISTRY.register
def resnet110(**kwargs):
    model = ResNet(depth=110, block=BasicBlock, **kwargs)
    return model


@MODEL_REGISTRY.register
def preact_resnet20(**kwargs):
    model = PreActResNet(depth=20, block=PreActBasicBlock, **kwargs)
    return model


@MODEL_REGISTRY.register
def preact_resnet32(**kwargs):
    model = PreActResNet(depth=32, block=PreActBasicBlock, **kwargs)
    return model


@MODEL_REGISTRY.register
def preact_resnet56(**kwargs):
    model = PreActResNet(depth=56, block=PreActBasicBlock, **kwargs)
    return model


@MODEL_REGISTRY.register
def preact_resnet110(**kwargs):
    model = PreActResNet(depth=110, block=PreActBasicBlock, **kwargs)
    return model
