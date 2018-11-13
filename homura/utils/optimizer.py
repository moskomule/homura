from abc import ABCMeta

from torch import nn
from torch.optim import SGD as _SGD, Adam as _Adam, RMSprop as _RMSprop


class Optimizer(metaclass=ABCMeta):

    def __init__(self, optim_cls, **kwargs):
        self._optim_cls = optim_cls
        self._args = kwargs
        self._optim = None

    def set_model(self, model: nn.Module):
        self._optim = self._optim_cls(model, **self._args)

    def step(self, closure=None):
        return self._optim.step(closure)

    def zero_grad(self):
        return self._optim.zero_grad()


class Adam(Optimizer):
    def __init__(self, **kwargs):
        super(Adam, self).__init__(_Adam, **kwargs)


class SGD(Optimizer):
    def __init__(self, **kwargs):
        super(SGD, self).__init__(_SGD, **kwargs)


class RMSProp(Optimizer):
    def __init__(self, **kwargs):
        super(RMSProp, self).__init__(_RMSprop, **kwargs)
