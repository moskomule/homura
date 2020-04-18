import random
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.transforms.functional import to_tensor


class PerChannelStatistics(object):
    """ estimates per channel image dataset statistics (mean, stdev) ::

        >>> estimator = PerChannelStatistics(100)
        >>> estimator.from_directory("data/images")
        >>> estimator.from_batch(img_tensor)
        >>> estimator.estimated
    """

    def __init__(self, num_samples: int):

        self._num_samples = num_samples
        self._mean = torch.zeros(3)
        self._stdev = torch.zeros(3)
        self._sample_count = 0

    def _calc(self, image: torch.Tensor):
        self._sample_count += 1
        flat_image = image.view(image.size(0), -1)
        self._mean += (flat_image.mean(dim=1) / self._num_samples)
        self._stdev += (flat_image.std(dim=1) / self._num_samples)

    def from_batch(self, batch: Sequence[torch.Tensor]):
        batch_size = len(batch)
        if batch_size < self._num_samples:
            raise RuntimeError(f"Need more than {self._num_samples} samples but {batch_size}")
        for image in batch[torch.randperm(batch_size).tolist()]:
            self._calc(image)
            if self._sample_count == self._num_samples:
                break
        return self.estimated

    def from_directory(self, root: Path or str):
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError

        image_paths = []
        for ext in IMG_EXTENSIONS:
            # *.jpg ...
            # in `root`
            image_paths += list(root.glob(f"*.{ext}"))
            # in subdirectories
            image_paths += list(root.glob(f"**/*.{ext}"))
            # *.JPG ...
            image_paths += list(root.glob(f"*.{ext.capitalize()}"))
            image_paths += list(root.glob(f"**/*.{ext.capitalize()}"))

        if len(image_paths) < self._num_samples:
            raise RuntimeError(f"Need more than {self._num_samples} samples but {len(image_paths)}")

        image_paths = random.sample(image_paths, k=self._num_samples)
        for path in image_paths:
            with path.open("rb") as f:
                img = Image.open(f).convert("RGB")
                self._calc(to_tensor(img))
            if self._sample_count == self._num_samples:
                break
        return self.estimated

    @property
    def mean(self):
        return self._mean

    @property
    def stdev(self):
        return self._stdev

    @property
    def estimated(self):
        return self._mean, self._stdev
