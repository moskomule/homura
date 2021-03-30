import pathlib

import numpy as np
from PIL import Image
from torchvision import datasets as VD


# to enable _split_dataset
def _svhn_getitem(self,
                  index: int):
    img, target = self.data[index], int(self.targets[index])
    img = Image.fromarray(np.transpose(img, (1, 2, 0)))
    if self.transform is not None:
        img = self.transform(img)
    return img, target


VD.SVHN.__getitem__ = _svhn_getitem


class ImageNet(VD.ImageFolder):
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 loader=None,
                 download=False):
        assert not download, "Download dataset by yourself!"
        root = pathlib.Path(root) / ('train' if train else 'val')
        kwargs = dict(transform=transform)
        if loader is not None:
            kwargs[loader] = loader
        super(ImageNet, self).__init__(root, **kwargs)
        import warnings

        warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


class OriginalSVHN(VD.SVHN):
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 download=False):
        super(OriginalSVHN, self).__init__(root, split="train" if train else "test", transform=transform,
                                           download=download)
        self.targets = self.labels


class ExtraSVHN(object):
    def __new__(cls,
                root,
                train=True,
                transform=None,
                download=False):
        if train:
            td = OriginalSVHN(root, train=True, transform=transform, download=download)
            ed = VD.SVHN(root, split='extra', transform=transform, download=download)
            td.data += ed.data
            td.targets += ed.labels
            return td
        else:
            return OriginalSVHN(root, train=False, transform=transform, download=download)

    def __len__(self):
        ...
