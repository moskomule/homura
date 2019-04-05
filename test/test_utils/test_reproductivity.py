import torch

from homura import set_seed
from homura.utils.reproductivity import unset_seed


def test_seed():
    set_seed()
    a = torch.randn(3, 3)
    b = torch.randn(4, 3)
    set_seed(0)
    assert torch.equal(a, torch.randn(3, 3))
    assert torch.equal(b, torch.randn(4, 3))

    unset_seed()
    assert not torch.equal(a, torch.randn(3, 3))
