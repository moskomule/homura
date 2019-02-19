import random
from abc import ABCMeta
from pathlib import Path
from typing import Iterable, Dict

import torch.utils.data as data
from PIL import Image


def _has_allowed_extension(file: Path, extensions: Iterable[str]):
    return file.suffix.lower() in extensions


def _find_classes(root: Path):
    classes = [d for d in root.iterdir() if d.is_dir()]
    classes.sort()
    class_to_idx = {d: i for i, d in enumerate(classes)}
    return classes, class_to_idx


def _make_dataset(root: Path, class_to_idx: Dict[str, int], extensions: Iterable[str]):
    images = []
    for d in [d for d in root.iterdir() if d.is_dir()]:
        for f in [f for f in d.iterdir() if _has_allowed_extension(f, extensions)]:
            images.append((d / f, class_to_idx[d]))
    return images


class FolderABC(data.Dataset, metaclass=ABCMeta):
    """
    Abstract class for ImageFolder and LabelCorruptedImages
    """

    def __init__(self, root, samples, transform=None, on_memory: bool = False):

        self.root = root
        self.samples = samples  # List of (image, target)
        self.length = len(self.samples)
        self.transforms = transform
        self.on_memory = on_memory

    def __getitem__(self, index):
        img, target = self.samples[index]
        if isinstance(img, Path):
            img = self.image_loader(img)
            if self.on_memory:
                self.samples[index] = (img, target)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def image_loader(path):
        with open(path, 'rb') as f:
            return Image.open(f).convert("RGB")


class ImageFolder(FolderABC):
    """A generic data loader where the images are arranged in this way ::

        root/cat/xxx.png
        root/dog/xxx.png

    :param root:
    :param transform:
    :param num_samples: Number of samples you want to use. If None, use all samples.
    :param on_memory: True if you want to store loaded images on the RAM for faster reloading. (False by default)
    """

    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

    def __init__(self, root,
                 transform=None,
                 num_samples: int = None,
                 on_memory: bool = False):
        classes, class_to_idx = _find_classes(root)
        samples = _make_dataset(root, class_to_idx, self.IMG_EXTENSIONS)
        if len(samples) == 0:
            raise RuntimeError(f"Found no image in subdirectories of {root}.")
        if num_samples is not None:
            if num_samples > len(samples):
                raise RuntimeError(f"Required too many samples ({num_samples}) but there are {len(samples)} samples")
            samples = random.sample(samples, k=num_samples)
        super(ImageFolder, self).__init__(root, samples, transform, on_memory)

        self.classes = classes
        self.class_to_index = class_to_idx
