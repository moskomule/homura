import numpy as np
import torch

from homura import set_deterministic, set_seed


def test_reproducibility():
    with set_seed(1):
        a = torch.randn(3, 3)
        b = torch.randn(4, 3)
        c = np.random.randn(3, 3)

    with set_seed(1):
        assert torch.equal(a, torch.randn(3, 3))
        assert torch.equal(b, torch.randn(4, 3))
        assert np.equal(c, np.random.randn(3, 3)).all()

    assert not torch.equal(a, torch.randn(3, 3))

    if not hasattr(torch, "set_deterministic"):
        with set_deterministic(0):
            assert not torch.backends.cudnn.benchmark
        assert torch.backends.cudnn.benchmark
