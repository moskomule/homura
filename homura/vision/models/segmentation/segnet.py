from torch import nn

__all__ = ["segnet"]


class Encoder(nn.Module):
    def __init__(self, in_channels, config):
        """
        encoder
        """
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.config = config
        self._make_layers(config)

    def forward(self, input):
        x = input
        sizes = tuple()
        indices = tuple()
        for i in range(len(self.config)):
            x = getattr(self, f"conv{i}")(x)
            sizes += (x.size(),)
            x, index = getattr(self, f"pool{i}")(x)
            indices += (index,)
        return x, indices, sizes

    def _make_layers(self, config):
        in_channels = self.in_channels
        for index, (out_channels, nums) in enumerate(config):
            convs = []
            for i in range(nums):
                convs += [nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                    padding=1),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(inplace=True)]
                in_channels = out_channels
            setattr(self, f"conv{index}", nn.Sequential(*convs))
            setattr(self, f"pool{index}", nn.MaxPool2d(
                kernel_size=2, stride=2, return_indices=True))


class Decoder(nn.Module):
    def __init__(self, encoder):
        """
        decoder
        """
        super(Decoder, self).__init__()
        self.encoder = encoder
        self._make_layers()

    def forward(self, input, indices, sizes):
        x = input
        for index in reversed(range(len(self.encoder.config))):
            x = getattr(self, f"unpool{index}")(
                x, indices[index], sizes[index])
            x = getattr(self, f"conv{index}")(x)
        return x

    def _make_layers(self):
        out_channels = self.encoder.in_channels  # e.g. 3
        config = self.encoder.config
        for index, (in_channels, nums) in enumerate(config):
            setattr(self, f"unpool{index}",
                    nn.MaxUnpool2d(kernel_size=2, stride=2))
            convs = []
            for i in range(nums):
                convs += [nn.ReLU(inplace=True),
                          nn.BatchNorm2d(out_channels),
                          nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
                out_channels = in_channels

            convs = list(reversed(convs))
            setattr(self, f"conv{index}", nn.Sequential(*convs))


class SegNet(nn.Module):
    config = ((64, 2),
              (128, 2),
              (256, 3),
              (512, 3),
              (512, 3))

    def __init__(self, num_classes, input_channels=3, use_imagenet=False):
        """
        SegNet, proposed in Badrinarayanan et al. (2015)
        :param num_classes: number of output classes
        :param input_channels: number of input channels
        >>> segnet = SegNet(10) # number of classes = 10
        >>> dummy = Variable(torch.randn(1, 3, 120, 120))
        >>> segnet(dummy).size() == (1, 10, 120, 120)
        """
        super(SegNet, self).__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.encoder = Encoder(self.input_channels, self.config)
        self.decoder = Decoder(self.encoder)
        self.output = nn.Conv2d(self.input_channels,
                                self.num_classes, kernel_size=3, padding=1)

        if use_imagenet:
            self._load_imagenet_weight()

    def forward(self, input):
        x = self.decoder(*self.encoder(input))
        x = self.output(x)
        return x

    def _load_imagenet_weight(self):
        from torchvision.models import vgg16_bn

        vgg = [value for key, value in vgg16_bn(pretrained=True).features.state_dict().items()
               if "running" not in key]
        encoder_convs = [m for m in self.encoder.modules() if isinstance(m, nn.Conv2d)]
        decoder_convs = [m for m in self.decoder.modules() if isinstance(m, nn.Conv2d)]
        for i, (e_m, d_m) in enumerate(zip(encoder_convs, decoder_convs)):
            weight, bias = vgg[4 * i: 4 * i + 2]
            e_m.load_state_dict({"weight": weight, "bias": bias})
            d_m.load_state_dict({"weight": weight, "bias": bias})


def segnet(num_classes, input_channels=3):
    return SegNet(num_classes, input_channels)
