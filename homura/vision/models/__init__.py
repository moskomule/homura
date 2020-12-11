from homura import Registry

MODEL_REGISTRY = Registry('vision_model')

from torchvision import models

MODEL_REGISTRY.register(models.resnet18)
MODEL_REGISTRY.register(models.resnet50)
MODEL_REGISTRY.register(models.wide_resnet50_2)
MODEL_REGISTRY.register(models.densenet121)
MODEL_REGISTRY.register(models.vgg19_bn)

from .densenet import densenet40, densenet100
from .cifar_resnet import wrn28_2, wrn40_2, wrn28_10, resnet20, resnet56, resnext29_32x4d

from .unet import unet
