"""
DenseNet for CIFAR dataset proposed in Gao et al. 2016
https://github.com/liuzhuang13/DenseNet
"""

import torch
from torch import nn
from torch.nn import functional as F

from homura.vision.models import MODEL_REGISTRY

__all__ = ["densenet40", "densenet100", "CIFARDenseNet"]

_padding = {"reflect": nn.ReflectionPad2d,
            "zero": nn.ZeroPad2d}


class _DenseLayer(nn.Module):

    def __init__(self, in_channels, bn_size, growth_rate, dropout_rate, padding):
        super(_DenseLayer, self).__init__()
        assert padding in _padding.keys()
        self.dropout_rate = dropout_rate
        self.layers = nn.Sequential(nn.BatchNorm2d(in_channels),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size=1, stride=1,
                                              bias=False),
                                    nn.BatchNorm2d(bn_size * growth_rate),
                                    nn.ReLU(inplace=True),
                                    _padding[padding](1),
                                    nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1,
                                              bias=False))

    def forward(self, input):
        x = self.layers(input)
        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        return torch.cat([input, x], dim=1)


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, bn_size, growth_rate, dropout_rate, padding):
        super(_DenseBlock, self).__init__()
        layers = [_DenseLayer(in_channels + i * growth_rate, bn_size, growth_rate, dropout_rate, padding)
                  for i in range(num_layers)]
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)


class _Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_Transition, self).__init__()
        self.layers = nn.Sequential(nn.BatchNorm2d(in_channels),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                                    nn.AvgPool2d(kernel_size=2, stride=2))

    def forward(self, input):
        return self.layers(input)


@MODEL_REGISTRY.register
class CIFARDenseNet(nn.Module):
    """
    DenseNet-BC (bottleneck and compactness) for CIFAR dataset. For ImageNet classification, use `torchvision`'s.

    :param num_classes: (int) number of output classes
    :param init_channels: (int) output channels which is performed on the input. 16 or 2 * growth_rate
    :param num_layers: (int) number of layers of each dense block
    :param growth_rate: (int) growth rate, which is referred as k in the paper
    :param dropout_rate: (float=0) dropout rate
    :param bn_size: (int=4) multiplicative factor in bottleneck
    :param reduction: (int=2) divisional factor in transition
    """

    def __init__(self, num_classes, init_channels, num_layers, growth_rate, dropout_rate=0, bn_size=4, reduction=2,
                 padding="reflect"):

        super(CIFARDenseNet, self).__init__()
        # initial conv.
        num_channels = init_channels
        layers = [_padding[padding](1), nn.Conv2d(3, num_channels, kernel_size=3, bias=False)]
        # first and second dense-block+transition
        for _ in range(2):
            layers.append(_DenseBlock(num_layers, in_channels=num_channels, bn_size=bn_size,
                                      growth_rate=growth_rate, dropout_rate=dropout_rate, padding=padding))
            num_channels = num_channels + num_layers * growth_rate
            layers.append(_Transition(num_channels, num_channels // reduction))
            num_channels = num_channels // reduction

        # third denseblock
        layers.append(_DenseBlock(num_layers, in_channels=num_channels, bn_size=bn_size, growth_rate=growth_rate,
                                  dropout_rate=dropout_rate, padding="reflect"))

        self.features = nn.Sequential(*layers)
        self.bn1 = nn.BatchNorm2d(num_channels + num_layers * growth_rate)
        self.linear = nn.Linear(num_channels + num_layers * growth_rate, num_classes)

        # initialize parameters
        self.initialize()

    def forward(self, input):
        x = self.features(input)
        x = F.relu(self.bn1(x), inplace=True)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return self.linear(x)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


def _cifar_densenet(depth, num_classes, growth_rate=12, **kwargs):
    n = (depth - 4) // 6
    model = CIFARDenseNet(num_classes, init_channels=2 * growth_rate, num_layers=n, growth_rate=growth_rate,
                          padding="reflect", **kwargs)
    return model


@MODEL_REGISTRY.register
def densenet100(num_classes, **kwargs):
    return _cifar_densenet(100, num_classes, **kwargs)


@MODEL_REGISTRY.register
def densenet40(num_classes, **kwargs):
    return _cifar_densenet(40, num_classes, **kwargs)
