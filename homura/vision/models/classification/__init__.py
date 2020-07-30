from homura import Registry

MODEL_REGISTRY = Registry('vision_model')

from .densenet import cifar_densenet100
from .resnet import resnet20, preact_resnet20, resnet56, preact_resnet56
from .wideresnet import wrn28_10, wrn28_2
from torchvision import models

MODEL_REGISTRY.register(models.resnet18)
MODEL_REGISTRY.register(models.resnet50)
MODEL_REGISTRY.register(models.wide_resnet50_2)
