from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .ema import exponential_moving_average_
from .functional import custom_straight_through_estimator, k_nearest_neighbor as knn


class VQModule(nn.Module):
    """ Vector Quantization module used in VQ-VAE [van den Oord et al. 17]

    :param emb_dim:
    :param dict_size:
    :param update_dict_by_ema:
    :param momentum:
    :param epsilon:
    :param knn_backend:
    """

    def __init__(self,
                 emb_dim: int,
                 dict_size: int,
                 update_dict_by_ema: bool = True,
                 momentum: float = 0.99,
                 epsilon: float = 1e-5,
                 knn_backend="torch"):

        super(VQModule, self).__init__()

        self.emb_dim = emb_dim
        self.dict_size = dict_size
        self.update_dict_by_ema = update_dict_by_ema
        self.epsilon = epsilon
        self._knn_backend = knn_backend
        embed = torch.randn(dict_size, emb_dim)

        if self.update_dict_by_ema:
            assert 0 <= momentum <= 1
            self.gamma = momentum
            self.register_buffer("_track_num", torch.zeros(dict_size, 1))
            self.register_buffer("_track_enc", embed.clone())
            self.register_buffer("embed", embed)
        else:
            self.register_parameter("embed", nn.Parameter(embed))

    def forward(self,
                input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        flatten = self.flatten_input(input)  # `? img emb_dim`
        with torch.no_grad():
            _, ids = knn(self.embed, flatten, 1, "l2",
                         backend=self._knn_backend)  # `dict_size img emb_dim`, `? img emb_dim`
            _input_size = list(input.size())
            _input_size.pop(1)
            ids = ids.view(*_input_size)

        vqs = self.lookup(ids)
        if self.training and self.update_dict_by_ema:
            self._ema_update(flatten, ids)
        if input.dim() == 4:
            vqs = vqs.transpose(1, -1)
        return custom_straight_through_estimator(vqs, input), ids

    def flatten_input(self,
                      input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 4:
            return input.transpose(1, -1).reshape(-1, self.emb_dim)
        elif input.dim() == 2:
            return input
        else:
            raise NotImplementedError

    def _ema_update(self,
                    flatten: torch.Tensor,
                    ids: torch.Tensor):
        with torch.no_grad():
            ids = ids.view(-1, 1)
            # `? img dict_size`
            onehot_ids = ids.new_zeros([ids.size(0), self.dict_size], dtype=torch.float)
            onehot_ids.scatter_(1, ids, 1)
            # `dict_size`
            exponential_moving_average_(self._track_num, onehot_ids.sum(dim=0).view_as(self._track_num), self.gamma)
            # `dict_size img emb_dim`
            exponential_moving_average_(self._track_enc, onehot_ids.t().matmul(flatten), self.gamma)

            # following sonnet's implementation
            factor = 1 + (self.epsilon * self.dict_size) / self._track_num.sum()
            self.embed = self._track_enc * factor / (self._track_num + self.epsilon)

    def lookup(self,
               ids: torch.Tensor):
        return F.embedding(ids, self.embed)
