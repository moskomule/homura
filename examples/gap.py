import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50

from homura import callbacks, reporter, optim
from homura.utils.containers import Map
from homura.trainers import TrainerBase
from homura.vision.data import imagenet_loaders


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
            module += [nn.Conv2d(num_filters * mul, num_filters * mul * 2,
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
    def __init__(self, models: dict, optimizer: optim.Optimizer, callbacks: callbacks.Callback, noise, verb=True):
        assert set(models.keys()) == {"generator", "classifier"}
        super(Trainer, self).__init__(models, {"generator": optimizer, "classifier": None},
                                      loss_f=F.cross_entropy,
                                      callbacks=callbacks, verb=verb)
        self.generator = self.model["generator"]
        self.classifier = self.model["classifier"]
        self.mag_in = 10
        self.noise = noise.to(self.device)

    def iteration(self, data):
        input, target = data
        results = Map()
        original_output = self.classifier(input)
        delta = self.model(self.noise)
        delta = self.normalize_and_scale(delta, self.mag_in)
        recons = self.clamp(input + delta, input)
        adv_output = self.classifier(recons)
        loss = self.loss_f(adv_output, original_output.argmin(dim=1))
        if self.is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        results.update(dict(loss=loss,
                            output=original_output,
                            prediction=original_output.argmax(dim=1),
                            adv_prediction=adv_output.argmax(dim=1),
                            adv_output=adv_output), )
        return results

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


@callbacks.metric_callback_decorator
def fooling_rate(data):
    prediction, adv_prediction, = data["prediction"], data["adv_prediction"]
    with torch.no_grad():
        return (prediction != adv_prediction).float().mean().item()


@callbacks.metric_callback_decorator
def adv_accuracy(data):
    adv_prediction, target = data["adv_prediction"], data["data"][1]
    with torch.no_grad():
        return (adv_prediction == target).float().mean().item()


def main():
    train_loader, test_loader = imagenet_loaders(args.root, args.batch_size,
                                                 num_train_samples=args.batch_size * args.train_max_iter,
                                                 num_test_samples=args.batch_size * args.test_max_iter)
    pretrained_model = resnet50(pretrained=True)
    for p in pretrained_model.parameters():
        p.requires_grad = False
    pretrained_model.eval()

    generator = ResNetGenerator(3, 3, args.num_filters)
    generator.cuda()
    optimizer = optim.Adam(lr=args.lr, betas=(args.beta1, 0.999))
    trainer = Trainer({"generator": generator, "classifier": pretrained_model},
                      optimizer,
                      reporter.TensorboardReporter(
                          [adv_accuracy, fooling_rate, callbacks.AccuracyCallback(),
                           callbacks.LossCallback()], save_dir="results"),
                      noise=torch.randn(3, 224, 224).expand(args.batch_size, -1, -1, -1))
    for ep in range(args.epochs):
        trainer.train(train_loader)
        trainer.test(test_loader)


if __name__ == '__main__':
    import miniargs

    p = miniargs.ArgumentParser()
    p.add_str("root")
    p.add_int("--batch_size", default=32)
    p.add_float("--lr", default=0.0002)
    p.add_float("--beta1", default=0.5)
    p.add_int("--train_max_iter", default=150)
    p.add_int("--test_max_iter", default=150)
    p.add_int("--epochs", default=10)
    p.add_int("--num_filters", default=64)
    args = p.parse()
    main()
