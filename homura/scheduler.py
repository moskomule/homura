from abc import ABCMeta

from torch.optim.lr_scheduler import (StepLR as _StepLR, MultiStepLR as _MultiStepLR, LambdaLR as _LambdaLR,
                                      ExponentialLR as _ExponentialLR, CosineAnnealingLR as _CosineAnnealingLR,
                                      ReduceLROnPlateau as _ReduceLROnPlateau)

from homura.optimizer import Optimizer


class Scheduler(metaclass=ABCMeta):
    def __init__(self, schdlr_cls, **kwargs):
        self._schdlr_cls = schdlr_cls
        self._kwargs = kwargs
        self._schdlr = None

    def set_optimizer(self, optimizer: Optimizer):
        self._schdlr = self._schdlr_cls(optimizer, **self._kwargs)

    @property
    def scheduler(self):
        return self._schdlr


class StepLR(Scheduler):
    def __init__(self, **kwargs):
        super(StepLR, self).__init__(_StepLR, **kwargs)


class MultiStepLR(Scheduler):
    def __init__(self, **kwargs):
        super(MultiStepLR, self).__init__(_MultiStepLR, **kwargs)


class LambdaLR(Scheduler):
    def __init__(self, **kwargs):
        super(LambdaLR, self).__init__(_LambdaLR, **kwargs)


class ExponentialLR(Scheduler):
    def __init__(self, **kwargs):
        super(ExponentialLR, self).__init__(_ExponentialLR, **kwargs)


class CosineAnnealingLR(Scheduler):
    def __init__(self, **kwargs):
        super(CosineAnnealingLR, self).__init__(_CosineAnnealingLR, **kwargs)


class ReduceLROnPlateau(Scheduler):
    def __init__(self, **kwargs):
        super(ReduceLROnPlateau, self).__init__(_ReduceLROnPlateau, **kwargs)
