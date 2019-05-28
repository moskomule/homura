# optimizers for homura's trainer

from abc import ABCMeta
from typing import Iterable

import torch
from torch import optim as torch_optim
from torch.optim import Optimizer as _Optimizer

from .utils import _optimizers

__all__ = ["Optimizer", "Adam", "SGD", "ASGD", "RMSProp", "AdaBound"]


class Optimizer(metaclass=ABCMeta):

    def __init__(self, optim_cls, **kwargs):
        self._optim_cls = optim_cls
        self._args = kwargs
        self._optim = None

    def set_model(self, params: Iterable[torch.Tensor]):
        self._optim = self._optim_cls(params, **self._args)
        return self.optim

    @property
    def optim(self) -> _Optimizer:
        return self._optim


class Adam(Optimizer):
    def __init__(self, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False):
        super(Adam, self).__init__(torch_optim.Adam,
                                   lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

    __doc__ = torch_optim.Adam.__doc__


class SGD(Optimizer):
    def __init__(self, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        super(SGD, self).__init__(torch_optim.SGD, lr=lr, momentum=momentum, dampening=dampening,
                                  weight_decay=weight_decay,
                                  nesterov=nesterov)

    __doc__ = torch_optim.SGD.__doc__


class RMSProp(Optimizer):
    def __init__(self, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False):
        super(RMSProp, self).__init__(torch_optim.RMSprop, lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay,
                                      momentum=momentum, centered=centered)

    __doc__ = torch_optim.RMSprop.__doc__


class ASGD(Optimizer):
    def __init__(self, lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0):
        super(ASGD, self).__init__(torch_optim.ASGD, lr=lr, lambd=lambd, alpha=alpha, t0=t0, weight_decay=weight_decay)

    __doc__ = torch_optim.ASGD.__doc__


class AdaBound(Optimizer):
    def __init__(self, lr=1e-3, betas=(0.9, 0.999), final_lr=0.1, gamma=1e-3,
                 eps=1e-8, weight_decay=0, amsbound=False):
        super(AdaBound, self).__init__(_optimizers.AdaBound, lr=lr, betas=betas, final_lr=final_lr, gamma=gamma,
                                       eps=eps, weight_decay=weight_decay, amsbound=amsbound)

    __doc__ = _optimizers.AdaBound.__doc__
