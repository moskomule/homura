# lr_schedulers for homura's trainer

from abc import ABCMeta

from torch.optim.lr_scheduler import (StepLR as _StepLR, MultiStepLR as _MultiStepLR, LambdaLR as _LambdaLR,
                                      ExponentialLR as _ExponentialLR, CosineAnnealingLR as _CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts as _CosineAnnealingWarmRestarts,
                                      ReduceLROnPlateau as _ReduceLROnPlateau, _LRScheduler)
from torch.optim.optimizer import Optimizer

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

    __doc__ = _StepLR.__doc__


class MultiStepLR(LRScheduler):
    def __init__(self, milestones, gamma=0.1, last_epoch=-1):
        super(MultiStepLR, self).__init__(_MultiStepLR, milestones=milestones, gamma=gamma, last_epoch=last_epoch)

    __doc__ = _MultiStepLR.__doc__


class LambdaLR(LRScheduler):
    def __init__(self, lr_lambda, last_epoch=-1):
        super(LambdaLR, self).__init__(_LambdaLR, lr_lambda=lr_lambda, last_epoch=last_epoch)

    __doc__ = _LambdaLR.__doc__


class ExponentialLR(LRScheduler):
    def __init__(self, gamma, last_epoch=-1):
        super(ExponentialLR, self).__init__(_ExponentialLR, gamma=gamma, last_epoch=last_epoch)

    __doc__ = _ExponentialLR.__doc__


class CosineAnnealingLR(LRScheduler):
    def __init__(self, T_max, eta_min=0, last_epoch=-1):
        super(CosineAnnealingLR, self).__init__(_CosineAnnealingLR, T_max=T_max, eta_min=eta_min, last_epoch=last_epoch)

    __doc__ = _CosineAnnealingLR.__doc__


class CosineAnnealingWarmRestart(LRScheduler):
    def __init__(self, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        super(CosineAnnealingWarmRestart, self).__init__(_CosineAnnealingWarmRestarts, T_0=T_0, T_mult=T_mult,
                                                         eta_min=eta_min, last_epoch=last_epoch)

    __doc__ = _CosineAnnealingWarmRestarts.__doc__


class ReduceLROnPlateau(LRScheduler):
    def __init__(self, mode='min', factor=0.1, patience=10,
                 verbose=False, threshold=1e-4, threshold_mode='rel',
                 cooldown=0, min_lr=0, eps=1e-8):
        super(ReduceLROnPlateau, self).__init__(_ReduceLROnPlateau, mode=mode, factor=factor, patience=patience,
                                                verbose=verbose, threshold=threshold, threshold_mode=threshold_mode,
                                                cooldown=cooldown, min_lr=min_lr, eps=eps)

    __doc__ = _ReduceLROnPlateau.__doc__
