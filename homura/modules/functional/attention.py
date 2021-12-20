from __future__ import annotations

import torch
from torch.nn import functional as F


def kv_attention(query: torch.Tensor,
                 key: torch.Tensor,
                 value: torch.Tensor,
                 mask: torch.Tensor = None,
                 additive_mask: torch.Tensor = None,
                 training: bool = True,
                 dropout_prob: float = 0,
                 scaling: bool = True
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention using queries, keys and value

    :param query: `...JxM`
    :param key: `...KxM`
    :param value: `...KxM`
    :param mask: `...JxK`
    :param additive_mask:
    :param training:
    :param dropout_prob:
    :param scaling:
    :return: torch.Tensor whose shape of `...JxM`
    """

    if scaling:
        query /= (query.size(-1) ** 0.5)
    attn = torch.einsum('...jm,...km->...jk', query, key).softmax(dim=-1)
    if mask is not None:
        if mask.dim() < attn.dim():
            mask.unsqueeze_(0)
        attn = attn.masked_fill(mask == 0, 1e-9)
    if additive_mask is not None:
        attn += additive_mask
    if training and dropout_prob > 0:
        attn = F.dropout(attn, p=dropout_prob)

    return torch.einsum('...jk,...km->...jm', attn, value), attn
