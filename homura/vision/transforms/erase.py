from typing import Tuple
from numbers import Number
import random, math
import numpy as np
from PIL import Image


class RandomErase(object):
    min_ratio = 0.1

    def __init__(self, erase_prob: float, area_ratio: Tuple[float, float] or float,
                 aspect_ratio: Tuple[float, float] or float):
        """
        Random Erasing Data Augmentation https://arxiv.org/abs/1708.04896
        :param erase_prob: erasing probability
        :param area_ratio: erasing area ratio range. If float, (min(0.1, x), max(0.1, x)) is used
        :param aspect_ratio: erasing aspect ratio range. If float, (min(0.1, x), max(0.1, x)) is used
        """
        self.area_ratio = 0
        self.aspect_ratio = 0
        self._set_ratios(area_ratio, aspect_ratio)

        assert 0 <= erase_prob <= 1, f"aspect_ratio should be in [0, 1], but {erase_prob}"
        self.erase_prob = erase_prob

    def __call__(self, img: Image):
        """
        :param img: input image
        :return: randomly erased PIL.Image
        """
        if random.random() < self.erase_prob:
            while True:
                w, h = img.size
                c = len(img.getbands())
                erase_area = random.uniform(*self.area_ratio) * w * h
                erase_aspect = random.uniform(*self.aspect_ratio)
                erase_h, erase_w = int(math.sqrt(erase_area * erase_aspect)), int(math.sqrt(erase_area / erase_aspect))
                # np (h, w, c) -> Image w, h + c
                x = random.randint(0, w)
                y = random.randint(0, h)

                if (x + erase_w <= w) and (y + erase_h <= h):
                    erase_rectangle = np.random.randint(0, 256, size=(erase_h, erase_h, c), dtype=np.uint8)
                    img.paste(Image.fromarray(erase_rectangle),
                              (x, y))
                    break
        return img

    def _set_ratios(self, area_ratio, aspect_ratio):

        def meta(ratio, ratio_name):
            if isinstance(ratio, Number):
                assert 0 < ratio <= 1
                setattr(self, ratio_name, (min(self.min_ratio, area_ratio),
                                           max(self.min_ratio, area_ratio)))
            elif isinstance(ratio, tuple):
                assert (0, 0) < area_ratio < (1, 1) and ratio[0] < ratio[1]
                setattr(self, ratio_name, ratio)
            else:
                raise TypeError(f"{ratio_name} should be float or (float, float) "
                                f"but {type(ratio)} is given")

        meta(area_ratio, "area_ratio")
        meta(aspect_ratio, "aspect_ratio")
