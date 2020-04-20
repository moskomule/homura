from typing import Tuple, Optional

import torch
from torch import nn

from .functional.attention import kv_attention


class KeyValAttention(nn.Module):
    """ Key-value attention.

    :param scaling:
    :param dropout_prob:
    """

    def __init__(self,
                 scaling: bool = False,
                 dropout_prob: float = 0):
        super(KeyValAttention, self).__init__()
        self._scaling = scaling
        self._dropout = dropout_prob

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                additive_mask: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ See `functional.attention.kv_attention` for details

        :param query:
        :param key:
        :param value:
        :param mask:
        :param additive_mask:
        :return:
        """

        return kv_attention(query, key, value, mask, additive_mask,
                            self.training, self._dropout, self._scaling)
