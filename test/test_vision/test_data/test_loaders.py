from pathlib import Path

import pytest
from torchvision import transforms

from homura.vision import mnist_loaders, cifar10_loaders


@pytest.mark.skipif(not Path("~/.torch/data/mnist").expanduser().exists(), reason="To avoid downloading")
@pytest.mark.parametrize("val_size", [0, 1000])
def test_mnist_loaders(val_size):
    data_augmentation = [transforms.RandomCrop(28, padding=4),
                         transforms.RandomHorizontalFlip()]
    ret = mnist_loaders(128, val_size=val_size, data_augmentation=data_augmentation)
    expected = 2 if val_size == 0 else 3
    assert len(ret) == expected

    for data in ret[0]:
        data
        break


@pytest.mark.skipif(not Path("~/.torch/data/cifar10").expanduser().exists(), reason="To avoid downloading")
@pytest.mark.parametrize("val_size", [0, 1000])
def test_cifar10_loaders(val_size):
    data_augmentation = [transforms.RandomCrop(32, padding=4),
                         transforms.RandomHorizontalFlip()]
    ret = cifar10_loaders(128, val_size=val_size, data_augmentation=data_augmentation)
    expected = 2 if val_size == 0 else 3
    assert len(ret) == expected

    for data in ret[0]:
        data
        break
