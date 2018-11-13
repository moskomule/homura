from abc import ABCMeta

from torch import nn
from torch.optim import SGD as _SGD, Adam as _Adam, RMSprop as _RMSprop, ASGD as _ASGD


class Optimizer(metaclass=ABCMeta):

    def __init__(self, optim_cls, **kwargs):
        self._optim_cls = optim_cls
        self._args = kwargs
        self._optim = None

    def set_model(self, model: nn.Module):
        self._optim = self._optim_cls(model, **self._args)

    @property
    def optim(self):
        return self._optim


class Adam(Optimizer):
    def __init__(self, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False):
        super(Adam, self).__init__(_Adam, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)


class SGD(Optimizer):
    def __init__(self, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        super(SGD, self).__init__(_SGD, lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay,
                                  nesterov=nesterov)


class RMSProp(Optimizer):
    def __init__(self, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False):
        super(RMSProp, self).__init__(_RMSprop, lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay,
                                      momentum=momentum, centered=centered)


class ASGD(Optimizer):
    def __init__(self, lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0):
        super(ASGD, self).__init__(_ASGD, lr=lr, lambd=lambd, alpha=alpha, t0=t0, weight_decay=weight_decay)
