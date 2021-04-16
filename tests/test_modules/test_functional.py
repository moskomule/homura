import torch
from pytest import approx
from torch.nn import functional as F

from homura.modules import functional as HF


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


def test_custom_ste():
    fwd = torch.randn(3)
    fwd2 = fwd.clone()
    bwd = torch.randn(3, requires_grad=True)
    bwd2 = bwd.detach().clone().requires_grad_(True)
    x = HF.custom_straight_through_estimator(fwd, bwd)
    assert torch.equal(x, fwd)
    (x ** 2).sum().backward()
    x2 = fwd2 + (bwd2 - bwd2.detach())
    assert torch.equal(x2, fwd2)
    (x2 ** 2).sum().backward()
    print(x, x2)
    assert torch.equal(bwd.grad.data, bwd2.grad.data)


def test_semantic_hashing():
    from homura.modules.functional.discretizations import _saturated_sigmoid

    for _ in range(10):
        input = torch.randn(3, requires_grad=True)
        dummy = input.clone().detach().requires_grad_(True)
        HF.semantic_hashing(input, is_training=True).sum().backward()
        _saturated_sigmoid(dummy).sum().backward()
        assert all(input.grad == dummy.grad)


def test_cross_entropy():
    input = torch.randn(1, 10)
    target = torch.tensor([4]).long()
    onehot_target = torch.zeros(1, 10)
    onehot_target[0, 4] = 1
    output = HF.cross_entropy_with_softlabels(input, onehot_target)
    expected = F.cross_entropy(input, target)
    assert output.item() == approx(expected.item())


def test_knn():
    k = 5
    keys = torch.randn(10, 6)
    qu = torch.randn(20, 6)
    s, i = HF.k_nearest_neighbor(keys, qu, k, "l2")
    assert s.size() == torch.Size([20, k])
