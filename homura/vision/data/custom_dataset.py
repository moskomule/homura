import torch.utils.data as data
from torch import randperm
from torch._utils import _accumulate

__all__ = ["TransformableSubset", "transformable_random_split"]


class TransformableSubset(data.Subset):
    """ Subset with transform
    """

    def __init__(self, dataset, indices):
        super(TransformableSubset, self).__init__(dataset, indices)
        self.transforms = None
        self.target_transform = None

    def update_transforms(self, transforms, target_transform=None):
        self.transforms = transforms
        self.target_transform = target_transform

    def __getitem__(self, idx):
        item = self.dataset[self.indices[idx]]
        img, target = item
        if self.transforms is not None:
            img = self.transforms(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


def transformable_random_split(dataset, lengths):
    indices = randperm(sum(lengths))
    return [TransformableSubset(dataset, indices[offset - length:offset])
            for offset, length in zip(_accumulate(lengths), lengths)]
