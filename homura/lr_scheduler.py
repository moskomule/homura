import bisect
import math
import warnings
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


def MultiStepWithWarmup(warmup: int,
                        milestones: list[int],
                        gamma: float = 0.1,
                        last_epoch: int = -1):
    return partial(_lr_scheduler.LambdaLR,
                   lr_lambda=multistep_with_warmup(warmup, milestones, gamma),
                   last_epoch=last_epoch)


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


def InverseSquareRootWithWarmup(warmup_epochs: int,
                                last_epoch: int = -1):
    """ inverse square root with warmup: $\\sqrt{w} \\min(1/\\sqrt{e}, e/\\sqrt{e}^3)$, where $w$ is `warmup_epochs` and
    `e` is the current epoch

    """
    return partial(_lr_scheduler.LambdaLR,
                   lr_lambda=inverse_square_root_with_warmup(warmup_epochs),
                   last_epoch=last_epoch)


def CosineAnnealingWithWarmup(total_epochs: int,
                              warmup_epochs: int,
                              min_lr: float = 0,
                              last_epoch: int = -1):
    if last_epoch == 0:
        warnings.warn("last_epoch is set to 0, is it intended?", DeprecationWarning)
    return partial(_CosineAnnealingWithWarmup, **locals())


class _CosineAnnealingWithWarmup(_lr_scheduler._LRScheduler):
    def __init__(self,
                 optimizer,
                 total_epochs: int,
                 warmup_epochs: int,
                 min_lr: float = 0,
                 last_epoch: int = -1):
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        super(_CosineAnnealingWithWarmup, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        def _warmup(warmup_epochs: int):

            def f(epoch):
                return (epoch + 1) / warmup_epochs

            return f

        warmup = _warmup(self.warmup_epochs)
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * warmup(self.last_epoch) for base_lr in self.base_lrs]

        else:
            new_epoch = self.last_epoch - self.warmup_epochs
            return [self.min_lr + (base_lr - self.min_lr) *
                    (1 + math.cos(math.pi * new_epoch / (self.total_epochs - self.warmup_epochs))) / 2
                    for base_lr in self.base_lrs]


def multistep_with_warmup(warmup_epochs: int,
                          milestones: list[int],
                          gamma: float = 0.1,
                          ):
    def f(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return gamma ** bisect.bisect_right(milestones, epoch)

    return f


def inverse_square_root_with_warmup(warmup_epochs: int,
                                    ):
    def f(epoch):
        epoch += 1
        factor = warmup_epochs ** 0.5
        return factor * min(epoch ** -0.5, epoch * warmup_epochs ** -1.5)

    return f
