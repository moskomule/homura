import torch
import pytest
from homura.modules.zca import ZCA


def test_zca():
    input = torch.randn(4, 3, 24, 24)
    module = ZCA.create(input)
    module(input)

    with pytest.raises(RuntimeError):
        module(torch.randn(3, 24, 24))
