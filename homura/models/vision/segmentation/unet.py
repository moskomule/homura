import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
import math


__all__ = ["unet"]


class Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        """
        >>> a = Variable(torch.randn(1, 1, 128, 128))
        >>> encoder = Block(1, 64)
        >>> encoder(a).size()
        torch.Size([1, 64, 128, 128])
        """
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(out_channel),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(out_channel, out_channel,
                                             kernel_size=3, padding=1),
                                   nn.BatchNorm2d(out_channel),
                                   nn.ReLU(inplace=True))

    def forward(self, input):
        return self.block(input)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channel, out_channel, upsample=True, block=Block):
        """
        >>> a = Variable(torch.randn(1, 1, 128, 128))
        >>> encoder = Block(1, 64)
        >>> encoder(a).size()
        torch.Size([1, 64, 128, 128])
        """
        super().__init__()
        if upsample:
            self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear"),
                                          nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1))
        else:
            self.upsample = nn.ConvTranspose2d(
                in_channel, out_channel, kernel_size=2)
        self.decoder = block(in_channel, out_channel)

    def forward(self, input, bypass):
        x = self.upsample(input)
        _, _, i_h, i_w = x.shape
        _, _, b_h, b_w = bypass.shape
        pad = (math.ceil((b_w - i_w) / 2), math.floor((b_w - i_w) / 2),
               math.ceil((b_h - i_h) / 2), math.floor((b_h - i_h) / 2))
        x = F.pad(x, pad)
        x = self.decoder(torch.cat([x, bypass], dim=1))
        return x


class UNet(nn.Module):
    def __init__(self, num_classes, input_channels, block=Block):
        """
        UNet, proposed in Ronneberger et al. (2015)
        :param num_classes: number of output classes
        :param input_channels: number of input channels
        >>> unet = UNet(10) # number of classes = 10
        >>> dummy = Variable(torch.randn(1, 3, 128, 128))
        >>> unet(dummy).shape
        torch.Size([1, 10, 128, 128])
        >>> dummy = Variable(torch.randn(1, 3, 427, 640))
        >>> unet(dummy).shape
        torch.Size([1, 10, 427, 640])
        """
        super().__init__()
        self.enc1 = block(input_channels, 64)
        self.enc2 = block(64, 128)
        self.enc3 = block(128, 256)
        self.enc4 = block(256, 512)
        self.enc5 = block(512, 1024)
        self.dec4 = UpsampleBlock(1024, 512, block=block)
        self.dec3 = UpsampleBlock(512, 256, block=block)
        self.dec2 = UpsampleBlock(256, 128, block=block)
        self.dec1 = UpsampleBlock(128, 64, block=block)
        self.down_conv1 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.loss = nn.NLLLoss2d()

        self.init_parameters()

    def forward(self, input):
        x1 = self.enc1(input)
        x2 = self.enc2(F.max_pool2d(x1, 2, 2))
        x3 = self.enc3(F.max_pool2d(x2, 2, 2))
        x4 = self.enc4(F.max_pool2d(x3, 2, 2))
        x = self.enc5(F.max_pool2d(x4, 2, 2))
        x = self.dec4(x, x4)
        x = self.dec3(x, x3)
        x = self.dec2(x, x2)
        x = self.dec1(x, x1)
        return self.down_conv1(x)

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def unet(num_classes, input_channels=3):
    return UNet(num_classes, input_channels)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
