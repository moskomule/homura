from functools import partial

import torch


def Adam(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, multi_tensor: bool = False):
    locs = locals()
    locs.pop("multi_tensor")
    opt = torch.optim._multi_tensor.Adam if multi_tensor else torch.optim.Adam
    return partial(opt, **locs)


def SGD(lr=1e-1, momentum=0, dampening=0, weight_decay=0, nesterov=False, multi_tensor: bool = False):
    locs = locals()
    locs.pop("multi_tensor")
    opt = torch.optim._multi_tensor.SGD if multi_tensor else torch.optim.SGD
    return partial(opt, **locs)


def RMSprop(lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False, multi_tensor: bool = False):
    locs = locals()
    locs.pop("multi_tensor")
    opt = torch.optim._multi_tensor.RMSprop if multi_tensor else torch.optim.RMSprop
    return partial(opt, **locs)


def AdamW(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False, multi_tensor: bool = False):
    locs = locals()
    locs.pop("multi_tensor")
    opt = torch.optim._multi_tensor.AdamW if multi_tensor else torch.optim.AdamW
    return partial(opt, **locs)
