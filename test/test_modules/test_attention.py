import pytest
import torch

from homura.modules.attention import KeyValAttention


@pytest.mark.parametrize("B", [4, 8])
@pytest.mark.parametrize("X", [0, 8])
@pytest.mark.parametrize("L", [1, 8])
@pytest.mark.parametrize("Y", [1, 8])
@pytest.mark.parametrize("K", [8, 9])
def test_attention(B, X, L, Y, K):
    queries = torch.randn(B, X, L)
    keys = torch.randn(B, Y, L)
    values = torch.randn(B, Y, K)
    mask = torch.randn(B, Y, L)
    attention = KeyValAttention()
    for m in (mask, None):
        feature_maps, attention_maps = attention(queries, keys, values, mask=m)
        assert feature_maps.size() == torch.Size((B, X, K))
        assert attention_maps.size() == torch.Size((B, X, Y))
