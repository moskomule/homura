import pathlib

from torchvision import datasets as VD


class ExtendedVOCSegmentation(object):
    def __new__(cls,
                root,
                train=True,
                transform=None,
                download=False):
        root = pathlib.Path(root).parent
        if train:
            return VD.SBDataset(root / "sbd", image_set='train_noval', mode='segmentation', transforms=transform,
                                download=download)
        else:
            return VD.VOCSegmentation(root / "voc", image_set="val", transforms=transform, download=download)

    def __len__(self):
        ...


# from torchvision's reference

def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


def seg_collate_fn(batch):
    images, targets = list(zip(*batch))
    batched_imgs = cat_list(images, fill_value=0)
    batched_targets = cat_list(targets, fill_value=255)
    return batched_imgs, batched_targets
