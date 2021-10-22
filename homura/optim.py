from functools import partial

import torch
from torch.optim import Optimizer


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


class LARC(object):
    """ LARC based on NVIDIA's Apex for Layer-wise Adaptive Rate Scaling. LARC is designed to wrap a given optimizer.
    Optimizer should be wrapped after initializing scheduler.
    """

    def __init__(self,
                 optimizer: Optimizer,
                 trust_coefficient: float = 0.02,
                 no_clip: bool = False,
                 eps: float = 1e-8):
        self.optim = optimizer
        self.trust_coefficient = trust_coefficient
        self.clip = not no_clip
        self.eps = eps

    def __getstate__(self):
        return self.optim.__getstate__()

    def __setstate__(self, state):
        self.optim.__setstate__(state)

    @property
    def state(self):
        return self.optim.state

    def __repr__(self):
        return self.optim.__repr__()

    @property
    def param_groups(self):
        return self.optim.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.optim.param_groups = value

    def state_dict(self):
        return self.optim.state_dict()

    def load_state_dict(self, state_dict):
        self.optim.load_state_dict(state_dict)

    def zero_grad(self):
        self.optim.zero_grad()

    def add_param_group(self, param_group):
        self.optim.add_param_group(param_group)

    @torch.no_grad()
    def step(self):
        weight_decays = []
        for group in self.optim.param_groups:
            # absorb weight decay control from optimizer
            weight_decay = group['weight_decay'] if 'weight_decay' in group else 0
            weight_decays.append(weight_decay)
            group['weight_decay'] = 0
            params = []
            grads = []
            lrs = []

            for p in group['params']:
                if p.grad is None:
                    continue
                param_norm = torch.norm(p.data)
                grad_norm = torch.norm(p.grad.data)

                if param_norm != 0 and grad_norm != 0:
                    # calculate adaptive lr + weight decay
                    # .item() may be sub-optimal, but required because _foreach_* don't support broadcasting at the moment
                    adaptive_lr = (self.trust_coefficient * param_norm /
                                   (grad_norm + param_norm * weight_decay + self.eps)).item()

                    # clip learning rate for LARC
                    if self.clip:
                        # calculation of adaptive_lr so that when multiplied by lr it equals `min(adaptive_lr, lr)`
                        adaptive_lr = min(adaptive_lr / group['lr'], 1.0)

                    params.append(p.data)
                    grads.append(p.grad.data)
                    lrs.append(adaptive_lr)

                    # p.grad.data += weight_decay * p.data
                    # p.grad.data *= adaptive_lr
            torch._foreach_add_(grads, params, alpha=weight_decay)
            torch._foreach_mul_(grads, lrs)

        self.optim.step()
        # return weight decay control to optimizer
        for i, group in enumerate(self.optim.param_groups):
            group['weight_decay'] = weight_decays[i]
