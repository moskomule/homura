from homura import Registry

MODEL_REGISTRY = Registry('vision_model')

from torchvision import models

MODEL_REGISTRY.register(models.resnet18)
MODEL_REGISTRY.register(models.resnet50)
MODEL_REGISTRY.register(models.resnet101)
MODEL_REGISTRY.register(models.resnet152)
MODEL_REGISTRY.register(models.wide_resnet50_2)
MODEL_REGISTRY.register(models.densenet121)
MODEL_REGISTRY.register(models.vgg19_bn)
if hasattr(models, "efficientnet_b0"):
    # >=1.10
    for v in range(8):
        MODEL_REGISTRY.register(getattr(models, f"efficientnet_b{v}"))

from .densenet import densenet40, densenet100
from .cifar_resnet import wrn28_2, wrn40_2, wrn28_10, resnet20, resnet56, resnext29_32x4d

from .unet import unet
