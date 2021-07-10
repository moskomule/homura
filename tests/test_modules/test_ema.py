import torch

from homura.modules.ema import EMA


def test_ema():
    model = torch.nn.Linear(3, 4)
    ema_model = EMA(model)
    input = torch.randn(5, 3)

    out = ema_model(input)
    assert out.shape == torch.Size([5, 4])
    assert not ema_model.ema_model.weight.requires_grad
    ema_model.requires_grad_(False)
    assert not ema_model.original_model.weight.requires_grad
