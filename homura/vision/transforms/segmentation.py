""" Transforms for segmentation task.

"""

import random
from typing import Optional

import numpy as np
import torch
from PIL import Image
from PIL.Image import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img: Image,
                   size: int,
                   fill: float = 0) -> Image:
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(object):
    def __init__(self,
                 transforms: list):
        self.transforms = transforms

    def __call__(self,
                 image: Image,
                 target: Image
                 ) -> (torch.Tensor, torch.Tensor):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomResize(object):
    def __init__(self,
                 min_size: int,
                 max_size: Optional[int] = None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self,
                 image: Image,
                 target: Image
                 ) -> (Image, Image):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=Image.NEAREST)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self,
                 flip_prob: float = 0.5):
        self.flip_prob = flip_prob

    def __call__(self,
                 image: Image,
                 target: Image
                 ) -> (Image, Image):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomCrop(object):
    def __init__(self,
                 size):
        self.size = size

    def __call__(self,
                 image: Image,
                 target: Image
                 ) -> (Image, Image):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self,
                 image: Image,
                 target: Image
                 ) -> (Image, Image):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class ToTensor(object):
    def __call__(self,
                 image: Image,
                 target: Image
                 ) -> (torch.Tensor, torch.Tensor):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.asarray(target), dtype=torch.int64)
        return image, target


class Normalize(object):
    def __init__(self,
                 mean: list or tuple,
                 std: list or tuple):
        self.mean = mean
        self.std = std

    def __call__(self,
                 image,
                 target
                 ) -> (torch.Tensor, torch.Tensor):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
