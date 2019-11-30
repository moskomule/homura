import copy
from typing import Optional

import torch
from torch import nn


def exponential_moving_average_(base: torch.Tensor,
                                update: torch.Tensor,
                                momentum: float) -> torch.Tensor:
    """ Inplace exponential moving average of `base` tensor

    :param base: tensor to be updated
    :param update: tensor for updating
    :param momentum:
    :return: exponential-moving-averaged `base` tensor
    """

    return base.mul_(momentum).add_(1 - momentum, update)


class EMANet(object):
    """ Track exponential moving average of a given model
        
    :param model: model to be tracked
    :param momentum:
    :param weight_decay:
    """

    def __init__(self,
                 model: nn.Module,
                 momentum: float,
                 weight_decay: Optional[float] = None):

        self.model = model
        self.ema_model = copy.deepcopy(model)
        self.ema_model.requires_grad_(False)
        self.momentum = momentum
        self.weight_decay = weight_decay

    def train(self):
        self.ema_model.train()

    def eval(self):
        self.ema_model.eval()

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self,
                        state_dict: dict):
        self.ema_model.load_state_dict(state_dict)

    def __call__(self,
                 input: torch.Tensor) -> torch.Tensor:
        return self.ema_model.forward(input)

    def update(self,
               update_buffers: bool = False):
        """ Update weights from the original model

        :param update_buffers: Update if buffers (e.g., running stats of BNs)
        """

        if update_buffers:
            for o_b, e_b in zip(self.model.buffers(), self.ema_model.buffers()):
                e_b.data.copy_(o_b.data)

        for o_p, e_p in zip(self.model.parameters(), self.ema_model.parameters()):
            exponential_moving_average_(e_p.data, o_p.data, self.momentum)
            if self.weight_decay is not None:
                o_p.data.mul_(1 - self.weight_decay)
