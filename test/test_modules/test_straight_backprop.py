import torch
from homura.modules import straight_backprop


def test_straight_backprop():
    a = torch.randn(4, 4, requires_grad=True)
    relu = straight_backprop(torch.relu)
    relu(a).sum().backward()
    assert (a.grad == a.new_ones(a.size())).all()
