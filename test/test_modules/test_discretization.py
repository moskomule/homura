import torch

from homura.modules import discretization


def test_gumbel_softmax():
    input = torch.randn(4, 10, 32, 32)
    gs = discretization.GumbelSoftmax(dim=1)
    assert gs(input).size() == input.size()
    gs.eval()
    assert gs(input).size() == input.size()
