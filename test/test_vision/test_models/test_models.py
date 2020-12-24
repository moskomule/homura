import pytest
import torch

from homura.vision.models import densenet100, resnet20, resnext29_32x4d, unet, wrn28_10


@pytest.mark.parametrize("num_classes", [10, 100])
@pytest.mark.parametrize("model", [resnet20, densenet100, resnext29_32x4d, wrn28_10])
def test_cifar(model, num_classes):
    input = torch.randn(2, 3, 32, 32)
    output = model(num_classes=num_classes)(input)
    assert output.size(1) == num_classes
    output.sum().backward()


def test_unet():
    input = torch.randn(1, 3, 224, 224)
    model = unet(num_classes=3)
    output = model(input)
    assert output.size() == torch.Size((1, 3, 224, 224))
    output.sum().backward()
