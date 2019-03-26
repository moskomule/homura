import torch

from homura.modules.miscs import straight_backprop, StraightBackprop


def test_straight_backprop():
    input = torch.randn(4, 4, requires_grad=True)
    relu = straight_backprop(torch.relu)
    relu(input).sum().backward()
    assert (input.grad == input.new_ones(input.size())).all()

    input = torch.randn(4, 4, requires_grad=True)
    relu = StraightBackprop(torch.relu)
    relu(input).sum().backward()
    assert (input.grad == input.new_ones(input.size())).all()
