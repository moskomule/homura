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

    return base.mul_(momentum).add_(update, alpha=1 - momentum)


class EMANet(nn.Module):
    """ Tracking exponential moving average of a given model

    :param model: model to be tracked
    :param momentum: momentum for EMA
    :param wrap_model: If True, `forward` returns outputs of the original model during training.
     If False, it returns outputs of the EMAed model. Set False if you need a pair of models.
    :param weight_decay: If a float value is given, apply weight decay to the original model.
    """

    def __init__(self,
                 model: nn.Module,
                 momentum: float,
                 wrap_model: bool = False,
                 weight_decay: Optional[float] = None):

        super(EMANet, self).__init__()
        self.original_model = model
        self.ema_model = copy.deepcopy(model)
        self.ema_model.requires_grad_(False)
        if momentum < 0 or 1 < momentum:
            raise RuntimeError(f"`momentum` is expected to be in [0, 1], but got {momentum}.")
        self.momentum = momentum
        self.wrap_model = wrap_model
        self.weight_decay = weight_decay

    def forward(self,
                *inputs: torch.Tensor,
                update_buffers: bool = True,
                **kwargs):

        if self.training:
            self.ema_model.train()
            model = self.original_model if self.wrap_model else self.ema_model
            outputs = model(*inputs, **kwargs)
            self._update(update_buffers)

        else:
            self.ema_model.eval()
            outputs = self.ema_model(*inputs, **kwargs)

        return outputs

    def _update(self,
                update_buffers: bool):
        # update ema model
        for o_p, e_p in zip(self.original_model.parameters(), self.ema_model.parameters()):
            exponential_moving_average_(e_p.data, o_p.data, self.momentum)
            if self.weight_decay is not None:
                o_p.data.mul_(1 - self.weight_decay)
        if update_buffers:
            for o_b, e_b in zip(self.original_model.buffers(), self.ema_model.buffers()):
                e_b.data.copy_(o_b.data)
