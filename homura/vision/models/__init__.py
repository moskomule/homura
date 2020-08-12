from homura import Registry

MODEL_REGISTRY = Registry('vision_model')

from torchvision import models

MODEL_REGISTRY.register(models.resnet18)
MODEL_REGISTRY.register(models.resnet50)
MODEL_REGISTRY.register(models.wide_resnet50_2)
MODEL_REGISTRY.register(models.densenet121)
MODEL_REGISTRY.register(models.vgg19_bn)

from .wideresnet import wrn28_2, wrn40_2, wrn28_10
from .densenet import densenet40, densenet100
from .resnet import resnet20, preact_resnet20, resnet56, preact_resnet56
from .resnext import resnext29_32x4d

from .unet import unet
