from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

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


class AttentionPool2d(nn.Module):
    # from openAI clip
    def __init__(self,
                 embed_dim: int,
                 num_heads: int):
        super().__init__()
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        x = x.flatten(-2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (1+HW)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x[0]
