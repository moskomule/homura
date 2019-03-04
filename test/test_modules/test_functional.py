import torch
from pytest import approx
from torch.nn import functional as F

from homura.modules import functional as HF


def test_gumbel_softmax():
    input = torch.tensor([10, 0.3, 0.3])
    samples = sum([HF.gumbel_softmax(input, 0, 0.01) for _ in range(400)]) / 400
    assert samples.tolist() == approx([1, 0, 0], abs=1e-2)


def test_gumbel_sigmoid():
    input = torch.tensor([10.0, -10.0])
    samples = sum([HF.gumbel_sigmoid(input, 0.01) for _ in range(400)]) / 400
    assert samples.tolist() == approx([1, 0], abs=1e-2)


def test_ste():
    input = torch.randn(3, requires_grad=True)
    dummy = input.clone().detach().requires_grad_(True)
    HF.straight_through_estimator(input).sum().backward()
    dummy.sum().backward()
    assert all(input.grad == dummy.grad)


def test_semantic_hashing():
    from homura.modules.functional.discretization import _saturated_sigmoid

    for _ in range(10):
        input = torch.randn(3, requires_grad=True)
        dummy = input.clone().detach().requires_grad_(True)
        HF.semantic_hashing(input, is_training=True).sum().backward()
        _saturated_sigmoid(dummy).sum().backward()
        assert all(input.grad == dummy.grad)


def test_cross_entropy():
    input = torch.randn(1, 10)
    target = torch.tensor([4]).long()
    onetho_target = torch.zeros(1, 10)
    onetho_target[0, 4] = 1
    output = HF.cross_entropy_with_softlabels(input, onetho_target)
    expected = F.cross_entropy(input, target)
    assert output.item() == approx(expected.item())
