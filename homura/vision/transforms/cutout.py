import random

import torch


def cutout(img: torch.Tensor,
           size: int,
           fill_value: float) -> torch.Tensor:
    _, h, w = img.size()
    mask = img.new_ones((1, h, w), dtype=torch.float32)
    x = random.randrange(w)
    y = random.randrange(h)
    x1 = min(w, max(x - size // 2, 0))
    x2 = min(w, max(x - size // 2, 0))
    y1 = min(h, max(y - size // 2, 0))
    y2 = min(h, max(y + size // 2, 0))
    mask[:, y1: y2, x1: x2] = fill_value
    img *= mask
    return img


class CutOut(object):
    """ Cutout for tensor

    """

    def __init__(self,
                 size: int,
                 fill_value: float = 0):
        self.size = size
        self.fill_value = fill_value

    def __call__(self,
                 img: torch.Tensor) -> torch.Tensor:
        return cutout(img, self.size, self.fill_value)
