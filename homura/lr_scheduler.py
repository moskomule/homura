# lr_schedulers for homura's trainer

from abc import ABCMeta

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (StepLR as _StepLR, MultiStepLR as _MultiStepLR, LambdaLR as _LambdaLR,
                                      ExponentialLR as _ExponentialLR, CosineAnnealingLR as _CosineAnnealingLR,
                                      ReduceLROnPlateau as _ReduceLROnPlateau, _LRScheduler)

__all__ = ["LRScheduler", "StepLR", "MultiStepLR", "LambdaLR", "ExponentialLR",
           "CosineAnnealingLR", "ReduceLROnPlateau"]


class LRScheduler(metaclass=ABCMeta):
    def __init__(self, schdlr_cls, **kwargs):
        self._schdlr_cls = schdlr_cls
        self._kwargs = kwargs
        self._schdlr = None

    def set_optimizer(self, optimizer: Optimizer):
        self._schdlr = self._schdlr_cls(optimizer, **self._kwargs)
        return self.scheduler

    @property
    def scheduler(self) -> _LRScheduler:
        return self._schdlr


class StepLR(LRScheduler):
    def __init__(self, step_size, gamma=0.1, last_epoch=-1):
        super(StepLR, self).__init__(_StepLR, step_size=step_size, gamma=gamma, last_epoch=last_epoch)


class MultiStepLR(LRScheduler):
    def __init__(self, milestones, gamma=0.1, last_epoch=-1):
        super(MultiStepLR, self).__init__(_MultiStepLR, milestones=milestones, gamma=gamma, last_epoch=last_epoch)


class LambdaLR(LRScheduler):
    def __init__(self, lr_lambda, last_epoch=-1):
        super(LambdaLR, self).__init__(_LambdaLR, lr_lambda=lr_lambda, last_epoch=last_epoch)


class ExponentialLR(LRScheduler):
    def __init__(self, gamma, last_epoch=-1):
        super(ExponentialLR, self).__init__(_ExponentialLR, gamma=gamma, last_epoch=last_epoch)


class CosineAnnealingLR(LRScheduler):
    def __init__(self, T_max, eta_min=0, last_epoch=-1):
        super(CosineAnnealingLR, self).__init__(_CosineAnnealingLR, T_max=T_max, eta_min=eta_min, last_epoch=last_epoch)


class ReduceLROnPlateau(LRScheduler):
    def __init__(self, mode='min', factor=0.1, patience=10,
                 verbose=False, threshold=1e-4, threshold_mode='rel',
                 cooldown=0, min_lr=0, eps=1e-8):
        super(ReduceLROnPlateau, self).__init__(_ReduceLROnPlateau, mode=mode, factor=factor, patience=patience,
                                                verbose=verbose, threshold=threshold, threshold_mode=threshold_mode,
                                                cooldown=cooldown, min_lr=min_lr, eps=eps)
