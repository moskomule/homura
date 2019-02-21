import torch.utils.data as data
from torch import randperm
from torch._utils import _accumulate


class TransformableSubset(data.Subset):
    """ Subset with transform
    """

    def __init__(self, dataset, indices):
        super(TransformableSubset, self).__init__(dataset, indices)
        self.transforms = None

    def update_transforms(self, transforms):
        self.transforms = transforms

    def __getitem__(self, idx):
        item = self.dataset[self.indices[idx]]
        if self.transforms is None:
            return item
        img, label = item
        return self.transforms(img), label


def transformable_random_split(dataset, lengths):
    indices = randperm(sum(lengths))
    return [TransformableSubset(dataset, indices[offset - length:offset])
            for offset, length in zip(_accumulate(lengths), lengths)]
