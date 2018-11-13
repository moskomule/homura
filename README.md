# Homura

*Homura* is a support tool for research experiments (for myself).

*Homura* (焰) is *flame* or *blaze* in Japanese. 🔥🔥🔥🔥

## Requirements

### minimal requirements

```
Python>=3.6
PyTorch==0.4.0
torchvision==0.2
```

```
yaml
tqdm
```

### optional

```
matplotlib
tensorboardX
visdom
```

## install

```console
pip install git+https://github.com/moskomule/homura#egg=homura
```

# Examples

See [examples](examples).

* [cifar10.py](examples/cifar10.py): training a CNN with random crop on CIFAR10
* [imagenet.py](examples/imagenet.py): training a CNN on ImageNet
* [gap.py](examples/gap.py): better implementation of generative adversarial perturbation