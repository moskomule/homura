import pytest
import torch
from torch import nn

from homura.modules import VQModule


@pytest.mark.parametrize("ema_update", [True, False])
@pytest.mark.parametrize("dim", [2, 4])
def test_vqmodule(ema_update, dim):
    if dim == 2:
        input = torch.randn(3, 4)
        f = nn.Linear(4, 10)
    else:
        input = torch.randn(3, 16, 4, 4)
        f = nn.Conv2d(16, 10, 3)
    vq = VQModule(10, 10, ema_update)
    for _ in range(2):
        ff = f(input)
        output, loss, ids = vq(ff)
        output.mean().backward()
        vq.zero_grad()
    if dim == 2:
        assert output.size() == ff.size()
        assert list(ids.size()) == [3]
    else:
        assert output.size() == ff.size()
        assert list(ids.size()) == [3, 2, 2]
