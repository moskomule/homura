import torch

from homura.vision import CutOut


def test_cutout():
    input = torch.randn(3, 32, 32)
    cutout = CutOut(16)
    cutout(input)
