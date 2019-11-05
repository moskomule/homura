from functools import partial

import torch


def Adam(lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
         weight_decay=0, amsgrad=False):
    return partial(torch.optim.Adam, **locals())


def SGD(lr=1e-1, momentum=0, dampening=0,
        weight_decay=0, nesterov=False):
    return partial(torch.optim.SGD, **locals())
