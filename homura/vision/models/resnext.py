import math
from typing import Callable, Optional, Type

import torch
from torch import nn

from . import MODEL_REGISTRY
from ._utils import conv1x1, conv3x3


class ResNeXtBottleneck(nn.Module):
    """ResNeXt Bottleneck Block type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)."""
    expansion = 4

    def __init__(self,
                 in_planes: int,
                 planes: int,
                 cardinality: int,
                 base_width: int,
                 stride: int,
                 norm: Optional[Type[nn.BatchNorm2d]],
                 act: Callable[[torch.Tensor], torch.Tensor]
                 ):
        super(ResNeXtBottleneck, self).__init__()

        dim = math.floor(planes * (base_width / 64))
        self.conv_reduce = conv1x1(in_planes, dim * cardinality, bias=norm is None)
        self.norm_reduce = nn.Identity() if norm is None else norm(dim * cardinality)
        self.conv_conv = conv3x3(dim * cardinality, dim * cardinality, stride=stride, groups=cardinality,
                                 bias=norm is None)
        self.norm = nn.Identity() if norm is None else norm(dim * cardinality)
        self.conv_expand = conv1x1(dim * cardinality, planes * 4, bias=norm is None)
        self.norm_expand = nn.Identity if norm is None else norm(planes * 4)

        self.act = act

        self.downsample = nn.Identity()
        if stride != 1 or in_planes != planes * self.expansion:
            _norm = nn.Identity() if norm is None else norm(planes * self.expansion)
            self.downsample = nn.Sequential(
                conv3x3(in_planes, planes * self.expansion, stride=stride, bias=norm is None),
                _norm
            )

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        bottleneck = self.conv_reduce(x)
        bottleneck = self.act(self.norm_reduce(bottleneck))

        bottleneck = self.conv_conv(bottleneck)
        bottleneck = self.act(self.norm(bottleneck))

        bottleneck = self.conv_expand(bottleneck)
        bottleneck = self.norm_expand(bottleneck)

        residual = self.downsample(x)
        return self.act(residual + bottleneck)


class CIFARResNeXt(nn.Module):

    def __init__(self,
                 num_classes: int,
                 depth: int,
                 cardinality: int,
                 base_width: int,
                 in_channels: int,
                 block: Type[ResNeXtBottleneck],
                 norm: Optional[Type[nn.BatchNorm2d]],
                 act: Callable[[torch.Tensor], torch.Tensor]
                 ):
        super(CIFARResNeXt, self).__init__()

        # Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        assert (depth - 2) % 9 == 0, 'depth should be one of 29, 38, 47, 56, 101'
        layer_blocks = (depth - 2) // 9
        width = 64

        self.cardinality = cardinality
        self.base_width = base_width
        self.num_classes = num_classes
        self.norm = norm
        self.act = act
        self.conv_1_3x3 = conv3x3(in_channels, width, bias=norm is None)
        self.norm_1 = nn.Identity() if norm is None else norm(width)

        self.inplanes = width
        self.stage_1 = self._make_layer(block, width, layer_blocks, 1)
        self.stage_2 = self._make_layer(block, 2 * width, layer_blocks, 2)
        self.stage_3 = self._make_layer(block, 4 * width, layer_blocks, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(4 * width * block.expansion, num_classes)

    def _make_layer(self, block: Type[ResNeXtBottleneck],
                    planes: int,
                    blocks: int,
                    stride: int
                    ) -> nn.Sequential:
        layers = [block(self.inplanes, planes, self.cardinality, self.base_width, stride, self.norm, self.act)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.cardinality, self.base_width, stride, self.norm, self.act))
        return nn.Sequential(*layers)

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        x = self.conv_1_3x3(x)
        x = self.act(self.norm_1(x))
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


@MODEL_REGISTRY.register
def resnext29_32x4d(num_classes=10):
    model = CIFARResNeXt(num_classes, depth=29, cardinality=4, base_width=32,
                         in_channels=3, block=ResNeXtBottleneck, norm=nn.BatchNorm2d, act=nn.ReLU())
    return model
