from homura import Registry

MODEL_REGISTRY = Registry('vision_model')

from torchvision import models

MODEL_REGISTRY.register(models.resnet18)
MODEL_REGISTRY.register(models.resnet50)
MODEL_REGISTRY.register(models.wide_resnet50_2)

from .wideresnet import wrn28_2, wrn40_2, wrn28_10
from .densenet import cifar_densenet100
from .resnet import resnet20, preact_resnet20

from .unet import unet
