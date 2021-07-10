import dataclasses

import torch

from homura.utils.containers import TensorTuple, TensorDataClass


def test_tensor_dataclass():
    @dataclasses.dataclass
    class TestClass(TensorDataClass):
        x: torch.Tensor
        y: torch.Tensor

    t = TestClass(torch.randn(3, 3), torch.randn(3, 3))
    x, y = t
    assert torch.equal(t.x, x)
    t_int = t.to(dtype=torch.int32)
    assert t_int.x.dtype == torch.int32


def test_tensortuple():
    a = torch.randn(3, 3), torch.randn(3, 3)
    t = TensorTuple(a)
    assert t[0].dtype == torch.float32

    assert t.to(torch.int32)[0].dtype == torch.int32
