import pytest
import torch
from homura.vision.data import PerChannelStatistics


@pytest.fixture
def input():
    return torch.randn(10, 3, 100, 100)


def test_estimation(input):
    estimator = PerChannelStatistics(10)
    mean, stdev = estimator.from_batch(input)
    # todo: is it ok?
    assert mean.mean().item() == pytest.approx(0, abs=1e-2)
    assert stdev.mean().item() == pytest.approx(1, abs=1e-2)
