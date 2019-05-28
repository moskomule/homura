import torch

from homura.vision.transforms.zca import zca_statistics, ZCATransformation


def test_zca():
    input = torch.randn(4, 3, 24, 24)
    zca_statistics(input, )

    zca_transform = ZCATransformation.create(input)
    assert zca_transform(input[0]).size() == input[0].size()
