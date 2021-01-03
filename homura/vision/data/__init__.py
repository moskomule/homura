from homura import Registry, get_environ
from .datasets import ImageNet, OriginalSVHN, VisionSet

DATASET_REGISTRY = Registry('vision_datasets', type=VisionSet)

from torchvision import datasets, transforms

DATASET_REGISTRY.register_from_dict(
    {
        'cifar10': VisionSet(datasets.CIFAR10, "~/.torch/data/cifar10", 10,
                             [transforms.ToTensor(),
                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))],
                             [transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                              transforms.RandomHorizontalFlip()]),

        'cifar100': VisionSet(datasets.CIFAR100, "~/.torch/data/cifar100", 100,
                              [transforms.ToTensor(),
                               transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))],
                              [transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                               transforms.RandomHorizontalFlip()]),

        'SVHN': VisionSet(OriginalSVHN, "~/.torch/data/svhn", 10,
                          [transforms.ToTensor(),
                           transforms.Normalize((0.4390, 0.4443, 0.4692), (0.1189, 0.1222, 0.1049))],
                          [transforms.RandomCrop(32, padding=4, padding_mode='reflect')]
                          ),

        'imagenet': VisionSet(ImageNet, get_environ('IMAGENET_ROOT', '~/.torch/data/imagenet'), 1_000,
                              [transforms.ToTensor(),
                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))],
                              [transforms.RandomResizedCrop(
                                  224), transforms.RandomHorizontalFlip()],
                              [transforms.Resize(256), transforms.CenterCrop(224)]),

    }
)
from .statistics import PerChannelStatistics
