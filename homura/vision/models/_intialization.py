from torch import nn


def init_parameters(module: nn.Module):
    """initialize parameters using kaiming normal"""
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            # todo: check if fan_out is valid
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
