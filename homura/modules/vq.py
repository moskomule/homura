from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


def moving_average_(base: torch.Tensor,
                    update: torch.Tensor,
                    momentum: float) -> torch.Tensor:
    """ Inplace exponential moving average of `base` tensor

    :param base:
    :param update:
    :param momentum:
    :return: exponential-moving-averaged `base` tensor
    """

    return base.mul_(momentum).add_(1 - momentum, update)


class VQModule(nn.Module):
    def __init__(self,
                 emb_dim: int,
                 dict_size: int,
                 update_dict_by_ema: bool = False,
                 gamma: float = 0.99,
                 epsilon: float = 1e-5):
        super(VQModule, self).__init__()
        self.emb_dim = emb_dim
        self.dict_size = dict_size
        self.update_dict_by_ema = update_dict_by_ema
        self.epsilon = epsilon
        embed = torch.randn(emb_dim, dict_size)

        if self.update_dict_by_ema:
            assert 0 <= gamma <= 1
            self.gamma = gamma
            self.register_buffer("_track_num", torch.zeros(1, dict_size))
            self.register_buffer("_track_enc", embed.clone())
            self.register_buffer("embed", embed)
        else:
            self.register_parameter("embed", nn.Parameter(embed))

    def forward(self,
                input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        flatten = self.flatten_input(input)

        with torch.no_grad():
            distance = VQModule.l2_distance(self.embed, flatten)
            # ids.size() == (?, 1)
            ids = distance.argmin(dim=1, keepdim=True)
            _input_size = list(input.size())
            _input_size.pop(1)
            ids = ids.view(*_input_size)

        vqs = self.lookup(ids)
        if self.training and self.update_dict_by_ema:
            self._ema(flatten, ids)
        if input.dim() == 4:
            vqs = vqs.transpose(1, -1)

        return vqs + input - input.detach(), ids

    @staticmethod
    def l2_distance(emb: torch.Tensor,
                    encoded: torch.Tensor) -> torch.Tensor:
        # emb: emb_dim x dict_size
        # encoded: ? x emb_dim
        # expected: ? x dict_size
        return (emb.pow(2).sum(dim=0, keepdim=True)
                + encoded.pow(2).sum(dim=1, keepdim=True)
                - 2 * encoded.matmul(emb))

    def flatten_input(self,
                      input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 4:
            return input.transpose(1, -1).reshape(-1, self.emb_dim)
        elif input.dim() == 2:
            return input
        else:
            raise NotImplementedError

    def _ema(self,
             flatten: torch.Tensor,
             ids: torch.Tensor):
        with torch.no_grad():
            # update of codebook is during backward
            # onehot_ids.size() == (?, dict_size)
            ids = ids.view(-1, 1)
            onehot_ids = ids.new_zeros([ids.size(0), self.dict_size], dtype=torch.float)
            onehot_ids.scatter_(1, ids, 1)
            moving_average_(self._track_num, onehot_ids.sum(dim=0).unsqueeze_(0), self.gamma)
            moving_average_(self._track_enc, flatten.t().matmul(onehot_ids), self.gamma)

            # following sonnet's implementation
            factor = 1 + (self.epsilon * self.dict_size) / self._track_num.sum()
            self.embed = self._track_enc * factor / (self._track_num + self.epsilon)

    def lookup(self,
               ids: torch.Tensor):
        return F.embedding(ids, self.embed.t())
