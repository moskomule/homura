import torch

from homura.vision.models import resnet20, preact_resnet20, cifar_densenet100, unet, wrn28_10


def test_resnet20():
    input = torch.randn(2, 3, 32, 32)
    model = resnet20(num_classes=10)
    output = model(input)
    assert output.size(1) == 10


def test_paresnet20():
    input = torch.randn(2, 3, 32, 32)
    model = preact_resnet20(num_classes=10)
    output = model(input)
    assert output.size(1) == 10


def test_densenet():
    input = torch.randn(2, 3, 32, 32)
    model = cifar_densenet100(num_classes=10)
    output = model(input)
    assert output.size(1) == 10


def test_wrn():
    input = torch.randn(2, 3, 32, 32)
    model = wrn28_10(num_classes=10)
    output = model(input)
    assert output.size(1) == 10


def test_unet():
    input = torch.randn(1, 3, 224, 224)
    model = unet(num_classes=3)
    assert model(input).size() == torch.Size((1, 3, 224, 224))
