from typing import Iterable, Dict
from pathlib import Path
import random
from PIL import Image

import torch
import torch.utils.data as data


def has_allowed_extension(file: Path, extensions: Iterable[str]):
    return file.suffix.lower() in extensions


def find_classes(root: Path):
    classes = [d for d in root.iterdir() if d.is_dir()]
    classes.sort()
    class_to_idx = {d: i for i, d in enumerate(classes)}
    return classes, class_to_idx


def make_dataset(root: Path, class_to_idx: Dict[str, int], extensions: Iterable[str]):
    images = []
    for d in [d for d in root.iterdir() if d.is_dir()]:
        for f in [f for f in d.iterdir() if has_allowed_extension(f, extensions)]:
            images.append((d / f, class_to_idx[d]))
    return images


class _DataSet(data.Dataset):

    def __init__(self, root, samples, transform=None, on_memory: bool = False):
        """
        Abstract class for ImageFolder and LabelCorruptedImages
        """

        self.root = root
        self.samples = samples
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


class ImageFolder(_DataSet):
    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

    def __init__(self, root, transform=None, on_memory: bool = False):
        """
        A generic data loader where the images are arranged in this way
            root/cat/xxx.png
            root/dog/xxx.png
        :param root:
        :param transform:
        :param on_memory: True if you want to store loaded images on the RAM for faster reloading. (False by default)
        """
        classes, class_to_idx = find_classes(root)
        samples = make_dataset(root, class_to_idx, self.IMG_EXTENSIONS)
        if len(samples) == 0:
            raise RuntimeError(f"Found 0 image in subdirectories of {root}.")
        super(ImageFolder, self).__init__(root, samples, transform, on_memory)

        self.classes = classes
        self.class_to_index = class_to_idx


class LabelCorruptedImages(ImageFolder):
    def __init__(self, root, transform, random_rate: float = 0, val_size: int = 0, random_seed: int = 6):
        """
        A subclass of ImageFloder whose labels are corrupted in given `random_rate`.
        >>>dataset = LabelCorruptedImages("here", random_rate=0.1, val_size=1500)
        >>>valset = dataset.valset()
        """
        super(LabelCorruptedImages, self).__init__(root, transform, on_memory=False)
        self.random_rate = random_rate
        self.val_size = val_size
        self.num_classes = len(self.classes)
        random.seed(random_seed)

        assert 0 <= random_rate <= 1
        if val_size > 0:
            original = range(len(self.samples))
            val_indices = random.sample(original, k=val_size)
            self._val_samples = [self.samples[i] for i in val_indices]
            new_indices = list(set(original).difference(val_indices))
            self.samples = [self.samples[i] for i in new_indices]

        if self.random_rate > 0:
            self._corrupt_labels()

    def _corrupt_labels(self):
        for idx, (img, target) in enumerate(self.samples):
            if random.random() < self.random_rate:
                classes = list(self.class_to_index.values())
                classes.remove(target)
                self.samples[idx] = (img, random.choice(classes))

    def valset(self, transform=None, on_memory=False):
        return _DataSet(root=self.root, samples=self._val_samples, transform=transform, on_memory=on_memory)


# THESE CODES ARE FROM THE PyTorch MASTER BRANCH #
class Subset(data.Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def random_split(dataset, lengths):
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = torch.randperm(sum(lengths))
    return [Subset(dataset, indices[offset - length:offset])
            for offset, length in zip(torch._utils._accumulate(lengths), lengths)]
