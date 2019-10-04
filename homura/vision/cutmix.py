import math
from typing import Optional, Tuple

import torch
import torch.jit

from .mixup import partial_mixup


def _clip(x: int, lower: int, upper: int) -> int:
    return min(max(x, lower), upper)


def rand_bbox(input: torch.Tensor,
              gamma: float) -> Tuple[int, int, int, int]:
    w = input.shape[2]
    h = input.shape[3]
    cut_rat = math.sqrt(1. - gamma)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)

    # uniform
    cx = int(torch.randint(0, w, (1,)))
    cy = int(torch.randint(0, h, (1,)))

    bbx1 = _clip(cx - cut_w // 2, 0, w)
    bby1 = _clip(cy - cut_h // 2, 0, h)
    bbx2 = _clip(cx + cut_w // 2, 0, w)
    bby2 = _clip(cy + cut_h // 2, 0, h)

    return bbx1, bby1, bbx2, bby2


def input_cutmix(input: torch.Tensor,
                 gamma: float,
                 indices: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """ CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features

    :param input:
    :param gamma:
    :param indices:
    :return:
    """

    bbx1, bby1, bbx2, bby2 = rand_bbox(input, gamma)
    input = input.clone()
    input[:, :, bbx1:bbx2, bby1:bby2] = input[indices, :, bbx1:bbx2, bby1:bby2]
    adjust_gamma = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
    return input, adjust_gamma


def cutmix(input: torch.Tensor,
           target: torch.Tensor,
           gamma: float,
           indices: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """ CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features

    :param input: The input tensor (B x C x H x W)
    :param target: The target tensor (B x C) is assumed to be onehot.
    :param gamma:
    :param indices:
    :return:
    """

    if indices is None:
        indices = torch.randperm(input.size(0), device=input.device, dtype=torch.long)

    input, gamma = input_cutmix(input, gamma, indices)
    return input, partial_mixup(target, gamma, indices)
