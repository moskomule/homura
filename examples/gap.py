import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50

from homura import callbacks, reporter, optim
from homura.data.folder import _DataSet, find_classes, make_dataset


class ImageFolder(_DataSet):
    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

    def __init__(self, root, transform=None, size=None, on_memory: bool = False):
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
        if size is not None:
            samples = random.sample(samples, k=size)
        super(ImageFolder, self).__init__(root, samples, transform, on_memory)

        self.classes = classes
        self.class_to_index = class_to_idx


class ResBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResBlock, self).__init__()
        module = []
        p = 0
        padding = {"reflect": nn.ReflectionPad2d,
                   "replicate": nn.ReplicationPad2d,
                   "zero": None}[padding_type]
        if padding is None:
            p = 1
        else:
            module.append(padding(1))

        module += [nn.Conv2d(dim, dim, 3, padding=p, bias=use_bias),
                   norm_layer(dim),
                   nn.ReLU(True)]
        if use_dropout:
            module.append(nn.Dropout(0.5))
        if padding is not None:
            module.append(padding(1))
        module += [nn.Conv2d(dim, dim, 3, padding=p, bias=use_bias),
                   norm_layer(dim)]
        self.module = nn.Sequential(*module)

    def forward(self, input):
        return input + self.module(input)


class ResNetGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, num_filters, norm="batchnorm",
                 activation="relu", use_dropout=False,
                 num_blocks=6, num_downsample=2, padding_type="reflect"):
        super(ResNetGenerator, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ngf = num_filters
        use_bias = (norm == "instance")
        norm_layer = {"batchnorm": nn.BatchNorm2d,
                      "instancenorm": nn.InstanceNorm2d}[norm]
        self.activation = {"selu": nn.SELU(True),
                           "relu": nn.ReLU(True)}[activation]
        module = [nn.ReflectionPad2d(3),
                  nn.Conv2d(self.in_channels, self.ngf, 7, bias=use_bias),
                  norm_layer(num_filters),
                  self.activation]
        for i in range(num_downsample):
            mul = 2 ** i
            module += [
                nn.Conv2d(num_filters * mul, num_filters * mul * 2,
                          kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(num_filters * mul * 2),
                self.activation]

        mul = 2 ** num_downsample
        for i in range(num_blocks):
            module += [ResBlock(num_filters * mul, padding_type, norm_layer, use_dropout, use_bias)]

        for i in range(num_downsample):
            mul = 2 ** (num_downsample - i)
            module += [nn.ConvTranspose2d(num_filters * mul, num_filters * mul // 2,
                                          kernel_size=3, stride=2, padding=1,
                                          output_padding=1, bias=use_bias),
                       norm_layer(num_filters * mul // 2),
                       self.activation]
        module += [nn.ReflectionPad2d(3),
                   nn.Conv2d(num_filters, self.out_channels, kernel_size=7, padding=0),
                   nn.Tanh()]
        self.module = nn.Sequential(*module)
        self.init_weight()

    def forward(self, input):
        return self.module(input)

    def init_weight(self):
        is_selu = isinstance(self.activation, nn.SELU)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if is_selu:
                    nn.init.normal_(m.weight, 0, 1 / math.sqrt(m.in_channels * m.kernel_size[0] * m.kernel_size[1]))
                else:
                    nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1, 0.02)
                nn.init.constant_(m.bias, 0)


class Trainer(TrainerBase):
    def __init__(self, generator: nn.Module, optimizer, callbacks, pretrained: nn.Module, noise: torch.Tensor):
        super(Trainer, self).__init__(generator, optimizer, loss_f=lambda x, y: F.cross_entropy(x, y).log(),
                                      callbacks=callbacks)
        self.pretrained = pretrained
        self.mag_in = 10
        self.noise = noise

    def iteration(self, data):
        if self.noise is None:
            raise RuntimeError
        input, target = self.to_device(data)
        original_output = self.pretrained(input)
        delta = self.model(self.noise)
        delta = self.normalize_and_scale(delta, self.mag_in)
        recons = self.clamp(input + delta, input)
        adv_output = self.pretrained(recons)
        loss = self.loss_f(adv_output, original_output.argmin(dim=1))
        if self.is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.register_after_iteration("prediction", original_output.argmax(dim=1))
        self.register_after_iteration("adv_prediction", adv_output.argmax(dim=1))

        return loss, original_output

    @staticmethod
    def clamp(recons, image):
        im = image.transpose(0, 1).reshape(image.size(1), -1)
        recons = torch.min(recons, im.max(dim=1, keepdim=True)[0].view(1, -1, 1, 1))
        recons = torch.max(recons, im.min(dim=1, keepdim=True)[0].view(1, -1, 1, 1))
        return recons

    mean_tensor = torch.Tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1).cuda()
    stddev_tensor = torch.Tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1).cuda()

    def normalize_and_scale(self, delta_im, mag_in):
        delta_im = delta_im + 1
        delta_im = delta_im * 0.5
        delta_im.sub_(self.mean_tensor).div_(self.stddev_tensor)
        b, c, *_ = delta_im.shape
        l_inf_channel = delta_im.view(b, c, -1).abs().max(dim=-1)[0]
        mag_in_scaled_c = (mag_in / (255 * self.stddev_tensor)).view(1, -1)
        delta_im = delta_im * torch.min(torch.ones_like(l_inf_channel),
                                        mag_in_scaled_c / l_inf_channel).view(b, c, 1, 1)
        return delta_im


class FoolingRate(callbacks.MetricCallback):
    def __init__(self):
        """
        calculate and accumulate accuracy
        :param k: top k accuracy (e.g. (1, 5) then top 1 and 5 accuracy)
        """
        super(FoolingRate, self).__init__(metric=self.fool, name="fooling_rate")

    @staticmethod
    def fool(data):
        prediction, adv_prediction, = data["prediction"], data["adv_prediction"]
        with torch.no_grad():
            return (prediction != adv_prediction).float().mean().item()


class AdvAccuracy(callbacks.MetricCallback):
    def __init__(self):
        super(AdvAccuracy, self).__init__(metric=self.accuracy, name="adv_accuracy")

    @staticmethod
    def accuracy(data):
        adv_prediction, target = data["adv_prediction"], data["data"][1]
        with torch.no_grad():
            return (adv_prediction == target).float().mean().item()


def data_loader(root, batch_size, train_size, test_size, num_workers=8):
    root = Path(root).expanduser()
    if not root.exists():
        raise FileNotFoundError
    train_size *= batch_size
    test_size *= batch_size

    _normalize = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    transform_base = [transforms.ToTensor(),
                      transforms.Normalize(*_normalize)]
    transform_test = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224)] + transform_base)
    train_loader = DataLoader(ImageFolder(root / "train", transform=transform_test, size=train_size),
                              batch_size=batch_size, num_workers=num_workers,
                              pin_memory=False, shuffle=True)
    test_loader = DataLoader(ImageFolder(root / "val", transform=transform_test, size=test_size),
                             batch_size=batch_size, num_workers=num_workers, pin_memory=False)
    return train_loader, test_loader


def get_noise(batch_size):
    noise = torch.rand(3, 224, 224)
    return torch.stack([noise for _ in range(batch_size)], dim=0)


def main():
    train_loader, test_loader = data_loader(args.root, args.batch_size, args.train_max_iter, args.test_max_iter)
    pretrained = Path(args.pretrained1)
    if not pretrained.exists():
        raise FileNotFoundError
    pretrained_model = resnet50()
    for p in pretrained_model.parameters():
        p.requires_grad = False
    pretrained_model.cuda()
    pretrained_model.eval()

    generator = ResNetGenerator(3, 3, args.num_filters)
    generator.cuda()
    optimizer = optim.Adam(lr=args.lr, betas=(args.beta1, 0.999))
    trainer = Trainer(generator, optimizer,
                      reporter.TensorboardReporter([FoolingRate(), AdvAccuracy(), callbacks.AccuracyCallback(),
                                                    callbacks.LossCallback()], save_dir="results"),
                      pretrained_model, get_noise(args.batch_size))
    for ep in range(args.epochs):
        trainer.train(train_loader)
        trainer.test(test_loader)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("pretrained")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.0002)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--train_max_iter", type=int, default=None)
    p.add_argument("--test_max_iter", type=int, default=None)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--num_filters", type=int, default=64)
    args = p.parse_args()
    main()
