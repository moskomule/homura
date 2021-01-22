import pytest
import torch

from homura.vision.transforms import transform as T

transforms = lambda: [T.RandomCrop(32, padding=4, padding_mode="reflect"),
                      T.RandomRotation(20),
                      T.RandomHorizontalFlip(),
                      T.RandomResizedCrop(24),
                      T.CenterCrop(24),
                      T.Normalize([0.5, 0.5, 0.5], [0.1, 0.1, 0.1]),
                      T.ColorJitter(0.1, 0.1, 0.1, 0.1),
                      T.RandomGrayScale()]


@pytest.mark.parametrize("transform", transforms())
def test_classification(transform):
    input = torch.randn(3, 32, 32)
    transform(input, target=None)


@pytest.mark.parametrize("transform", transforms())
def test_detection(transform):
    transform.target_type = "bbox"
    input = torch.randn(3, 64, 64)
    target = torch.tensor([[2, 4, 10, 14],
                           [30, 40, 50, 61]], dtype=torch.float)
    transform(input, target=target)


@pytest.mark.parametrize("transform", transforms())
def test_segmentation(transform):
    transform.target_type = "mask"
    input = torch.randn(3, 64, 64)
    target = torch.randint(8, (3, 64, 64))
    transform(input, target=target)


def test_concat():
    transform1 = T.RandomCrop(32, padding=4, padding_mode="reflect")
    transform2 = T.RandomRotation(30)
    assert str(T.ConcatTransform(transform2, transform1)) == str((transform1 * transform2))
    transform1 * transform2 * transform1
