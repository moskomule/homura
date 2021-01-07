import xml.etree.ElementTree as ET

import torch
from torchvision import datasets as VD
from torchvision.io import read_image


def det_collate_fn(batch):
    return tuple(zip(*batch))


class VOCDetection(object):
    def __new__(cls,
                root,
                train=True,
                transform=None,
                download=False):
        if train:
            train_kwargs = dict(image_set='trainval', transforms=transform, download=download)
            train_2007 = _VOCDetection(root, year='2007', **train_kwargs)
            train_2012 = _VOCDetection(root, year='2012', **train_kwargs)
            remove_difficult(train_2007)
            remove_difficult(train_2012)
            return train_2007 + train_2012

        else:
            val_set = _VOCDetection(root, year='2007', image_set='test', transforms=transform, download=download)
            return val_set

    def __len__(self):
        ...


class _VOCDetection(VD.VOCDetection):
    voc_bbox_label_names = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                            'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                            'train', 'tvmonitor')

    def __getitem__(self, item):

        img = read_image(self.images[item])
        tgt = self.parse_voc_xml(ET.parse(self.annotations[item]).getroot())
        bbox = []
        label = []
        difficult = []
        objs = tgt['annotation']['object']
        if not isinstance(objs, list):
            objs = [objs]
        for obj in objs:
            bbox.append([int(obj['bndbox'][k]) for k in ('xmin', 'ymin', 'xmax', 'ymax')])
            label.append(self.voc_bbox_label_names.index(obj['name']))
            difficult.append(int(obj['difficult']))
        tgt = dict(boxes=torch.tensor(bbox, dtype=torch.long),
                   labels=torch.tensor(label, dtype=torch.long),
                   difficult=torch.tensor(difficult, dtype=torch.long))
        if self.transforms is not None:
            bbox = tgt['boxes'].float()
            img, bbox = self.transforms(img, bbox)
            tgt['boxes'] = bbox.long()
        return img, tgt


def remove_difficult(dataset: _VOCDetection):
    for index in range(len(dataset)):
        target = dataset.parse_voc_xml(ET.parse(dataset.annotations[index]).getroot())
        objs = target['annotation']['object']
        if not isinstance(objs, list):
            objs = [objs]
        target['annotation']['object'] = [obj for obj in objs if obj['difficult'] != '1']

        if len(objs) == 0:
            dataset.images.pop(index)
            dataset.annotations.pop(index)
