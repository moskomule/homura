from typing import Optional

import torch
from torch.nn import functional as F


def kv_attention(query: torch.Tensor,
                 key: torch.Tensor,
                 value: torch.Tensor,
                 mask: Optional[torch.Tensor] = None,
                 additive_mask: Optional[torch.Tensor] = None,
                 training: bool = True,
                 dropout_prob: float = 0,
                 scaling: bool = True
                 ) -> (torch.Tensor, torch.Tensor):
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


def lsh_attention(query: torch.Tensor,
                  key: Optional[torch.Tensor],
                  value: torch.Tensor,
                  mask: Optional[torch.Tensor] = None,
                  ) -> (torch.Tensor, torch.Tensor):
    pass


def lsh(input: torch.Tensor,
        num_hashes: int,
        num_buckets: int
        ) -> torch.Tensor:
    # input: ...JxM
    # rot: Mx{num_hashes}x{num_buckets//2}
    rot = input.new_empty(input.size(-1), num_hashes, num_buckets // 2).normal_()
    rot_vec = torch.einsum("...jm,mhb->...hjb", input, rot)
    # ...MxHx{num_buckets}
    rot_vec = torch.cat([rot_vec, -rot_vec], -1)
    bucket_range = torch.arange(rot_vec.size(-1), device=rot_vec.device)
