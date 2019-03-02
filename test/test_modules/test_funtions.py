import torch
from pytest import approx
from torch.nn import functional as F

from homura.modules import functions


def test_straight_backprop():
    a = torch.randn(4, 4, requires_grad=True)
    relu = functions.straight_backprop(torch.relu)
    relu(a).sum().backward()
    assert (a.grad == a.new_ones(a.size())).all()


def test_gumbel_softmax():
    a = torch.tensor([10, 0.3, 0.3])
    samples = sum([functions.gumbel_softmax(a, 0, 0.01) for _ in range(400)]) / 400
    assert samples.tolist() == approx([1, 0, 0], abs=1e-2)


def test_gumbel_sigmoid():
    a = torch.tensor([10.0, -10.0])
    samples = sum([functions.gumbel_sigmoid(a, 0.01) for _ in range(400)]) / 400
    assert samples.tolist() == approx([1, 0], abs=1e-2)


def test_ste():
    input = torch.randn(3, requires_grad=True)
    dummy = input.clone().detach().requires_grad_(True)
    functions.straight_through_estimator(input).sum().backward()
    dummy.sum().backward()
    assert all(input.grad == dummy.grad)


def test_semantic_hashing():
    from homura.modules.functions.discretization import _saturated_sigmoid

    for _ in range(10):
        input = torch.randn(3, requires_grad=True)
        dummy = input.clone().detach().requires_grad_(True)
        functions.semantic_hashing(input, is_training=True).sum().backward()
        _saturated_sigmoid(dummy).sum().backward()
        assert all(input.grad == dummy.grad)


def test_cross_entropy():
    input = torch.randn(1, 10)
    target = torch.tensor([4]).long()
    onetho_target = torch.zeros(1, 10)
    onetho_target[0, 4] = 1
    output = functions.cross_entropy_with_softlabels(input, onetho_target)
    expected = F.cross_entropy(input, target)
    assert output.item() == approx(expected.item())
