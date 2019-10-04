import torch
import torch.jit

from homura.vision import cutmix


def test_cutmix():
    input = torch.empty(4, 3, 32, 32)
    target = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    cutmix(input, target, 0.1)

    jit_cutmix = torch.jit.script(cutmix)
    jit_cutmix(input, target, 0.1)
