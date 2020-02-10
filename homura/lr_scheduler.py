import math
from functools import partial

from torch.optim import lr_scheduler as _lr_scheduler


def StepLR(step_size,
           gamma=0.1,
           last_epoch=-1):
    return partial(_lr_scheduler.StepLR, **locals())


def MultiStepLR(milestones,
                gamma=0.1,
                last_epoch=-1):
    return partial(_lr_scheduler.MultiStepLR, **locals())


def LambdaLR(lr_lambda,
             last_epoch=-1):
    return partial(_lr_scheduler.LambdaLR, **locals())


def ExponentialLR(T_max,
                  eta_min=0,
                  last_epoch=-1):
    return partial(_lr_scheduler.ExponentialLR, **locals())


def ReduceLROnPlateau(mode='min',
                      factor=0.1,
                      patience=10,
                      verbose=False,
                      threshold=1e-4,
                      threshold_mode='rel',
                      cooldown=0,
                      min_lr=0,
                      eps=1e-8):
    return partial(_lr_scheduler.ReduceLROnPlateau, **locals())


def CosineAnnealingWithWarmup(total_epochs: int,
                              multiplier: float,
                              warmup_epochs: int,
                              min_lr: float = 0,
                              last_epoch: int = -1):
    return partial(_CosineAnnealingWithWarmup, **locals())


class _CosineAnnealingWithWarmup(_lr_scheduler._LRScheduler):
    def __init__(self,
                 optimizer,
                 total_epochs: int,
                 multiplier: float,
                 warmup_epochs: int,
                 min_lr: float = 0,
                 last_epoch: int = -1):
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.multiplier = multiplier
        self.warmup_epochs = warmup_epochs
        super(_CosineAnnealingWithWarmup, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        def _warmup(multiplier: float,
                    warmup_epochs: int):
            # Finally (at the warmup_epochs-th epoch), lr becomes base_lr

            assert multiplier >= 1
            mul = 1 / multiplier
            return lambda epoch: (1 - mul) * epoch / warmup_epochs + mul

        warmup = _warmup(self.multiplier, self.warmup_epochs)
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * warmup(self.last_epoch) for base_lr in self.base_lrs]

        else:
            new_epoch = self.last_epoch - self.warmup_epochs
            return [self.min_lr + (base_lr - self.min_lr) *
                    (1 + math.cos(math.pi * new_epoch / (self.total_epochs - self.warmup_epochs))) / 2
                    for base_lr in self.base_lrs]
