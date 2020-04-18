import pytest
import torch

from homura.modules.functional.attention import kv_attention


@pytest.mark.parametrize("B", [4, 8])
@pytest.mark.parametrize("J", [3, 8])
@pytest.mark.parametrize("M", [1, 8])
@pytest.mark.parametrize("K", [1, 8])
def test_attention(B, J, M, K):
    queries = torch.randn(B, J, M)
    keys = torch.randn(B, K, M)
    values = torch.randn(B, K, M)
    mask = torch.randn(J, K)
    for m in (mask, None):
        feature_maps, attention_maps = kv_attention(queries, keys, values, mask=m)
        assert feature_maps.size() == torch.Size((B, J, M))
        assert attention_maps.size() == torch.Size((B, J, K))
