import torch.utils.data as data
from torch import randperm
from torch._utils import _accumulate

__all__ = ["TransformableSubset", "transformable_random_split"]


class TransformableSubset(data.Subset):
    """ Subset with transform. The original `Subset` inherits transformations from the original Dataset,
    which may not be useful in some cases, e.g. train-validation split.
    """

    def __init__(self, dataset, indices):
        super(TransformableSubset, self).__init__(dataset, indices)
        self.transform = None
        self.target_transform = None

    def update_transforms(self, transform, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        item = self.dataset[self.indices[idx]]
        img, target = item
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


def transformable_random_split(dataset, lengths):
    indices = randperm(sum(lengths))
    return [TransformableSubset(dataset, indices[offset - length:offset])
            for offset, length in zip(_accumulate(lengths), lengths)]
