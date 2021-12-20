import torch

from homura.vision import mixup


def test_cutmix():
    input = torch.empty(4, 3, 32, 32)
    target = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    mixup(input, target, 0.1)
