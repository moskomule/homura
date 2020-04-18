from typing import Optional

import torch
from torch import nn

from homura.modules.attention import KeyValAttention


class MultiHeadAttention(nn.Module):
    _relative_pos_clip = 2

    def __init__(self,
                 embed_size: int,
                 num_heads: int = 8,
                 hidden_size: Optional[int] = None,
                 dropout_prob: float = 0,
                 relative_pos: bool = False
                 ):
        super(MultiHeadAttention, self).__init__()
        if hidden_size is None:
            hidden_size = embed_size
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.rel_pos_emb = nn.Embedding(self._relative_pos_clip * 2 + 1, self.hidden_size) if relative_pos else None
        self.q_emb = nn.Linear(self.embed_size, self.hidden_size)
        self.k_emb = nn.Linear(self.embed_size, self.hidden_size)
        self.v_emb = nn.Linear(self.embed_size, self.hidden_size)
        self.last_lin = nn.Linear(self.hidden_size, self.embed_size)
        self.attn = KeyValAttention(scaling=True, dropout_prob=dropout_prob)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None
                ) -> (torch.Tensor, torch.Tensor):
        """

        :param query: `...JxM`
        :param key: `...KxM`
        :param value: `...KxM`
        :param mask: `...JxK`
        :return:
        """

        bs = query.size(0)
        dim_per_head = self.hidden_size // self.num_heads
        shape = (bs, -1, self.num_heads, dim_per_head)

        # ...JxM -> ...JxH
        q = self.q_emb(query)
        rel_pos_logits = None
        if self.rel_pos_emb is not None:
            pos = torch.arange(key.size(-2), device=key.device)
            # JxK
            rel_pos = (pos - torch.arange(query.size(-2), device=key.device)[:, None]).clamp_(-self._relative_pos_clip,
                                                                                              self._relative_pos_clip)
            rel_pos += self._relative_pos_clip
            # 1xJxH * 1xJxK
            rel_pos_logits = (q.unsqueeze(-2) * self.rel_pos_emb(rel_pos).unsqueeze(0)).sum(dim=-1, keepdim=True)
        # -> {bs}x{self.num_heads}x{-1}x{dim_per_head}
        q = q.view(shape).transpose_(1, 2)
        k = self.k_emb(key).view(shape).transpose_(1, 2)
        v = self.v_emb(value).view(shape).transpose_(1, 2)
        feature, attn = self.attn(q, k, v, mask, additive_mask=rel_pos_logits)
        feature.transpose_(1, 2).reshape(bs, -1, self.hidden_size)
        return self.last_lin(feature), attn
