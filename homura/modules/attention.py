import math
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


class KeyValAttention(nn.Module):
    def __init__(self, scaling: bool = False, dropout_ratio: float = 0):
        super(KeyValAttention, self).__init__()
        self._scaling = scaling
        self._dropout = nn.Dropout(dropout_ratio) if dropout_ratio > 0 else None

    def forward(self,
                queries: torch.Tensor,
                keys: torch.Tensor,
                values: torch.Tensor,
                mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        dim of X >= 0,
        dim of Y >= 1

        :param queries: B x X x L
        :param keys: B x Y x L
        :param values: B x Y x L
        :param mask: B x Y x L or None
        :return:
        """
        raw_attention = queries @ keys.transpose(-2, -1)
        if self._scaling:
            # see Transformer
            raw_attention = raw_attention / math.sqrt(queries.shape[-1])
        if self._dropout is not None:
            mask = raw_attention.new_ones(raw_attention.size()) if mask is None else mask
            mask = self._dropout(mask)
            raw_attention = raw_attention.masked_fill(mask == 0, -1e9)
        attention_maps = F.softmax(raw_attention, dim=-1)
        feature_maps = attention_maps @ values

        return feature_maps, attention_maps
