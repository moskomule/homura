import pytest
import torch
from torch import nn

from homura import callbacks
from homura.callbacks import AccuracyCallback, TQDMReporter, IOReporter, TensorboardReporter
from homura.optim import SGD
from homura.trainers import SupervisedTrainer


def test_metrics_single_value():
    @callbacks.metric_callback_decorator
    def f(data):
        return data["test"]

    metric = f
    num_iter = 4
    metric.before_all({})
    metric.before_epoch({"mode": "train"})
    for _ in range(num_iter):
        metric.before_iteration({"mode": "train"})
        metric.after_iteration({"mode": "train", "test": 1})
    metric.after_epoch({"mode": "train", "iter_per_epoch": num_iter})
    metric.before_epoch({"mode": "val"})
    for _ in range(num_iter):
        metric.before_iteration({"mode": "val"})
        metric.after_iteration({"mode": "val", "test": 0.1})
    metric.after_epoch({"mode": "val", "iter_per_epoch": num_iter})

    assert pytest.approx(metric.history["train"][0] == 1)
    assert pytest.approx(metric.history["val"][0] == 0.1)


def test_metrics_multiple_values():
    @callbacks.metric_callback_decorator
    def f(data):
        return data["test"]

    metric = f
    num_iter = 4
    metric.before_all({})
    metric.before_epoch({"mode": "train"})
    for _ in range(num_iter):
        metric.before_iteration({"mode": "train"})
        metric.after_iteration({"mode": "train", "test": torch.tensor([1.0, 1.2])})
    metric.after_epoch({"mode": "train", "iter_per_epoch": num_iter})
    metric.before_epoch({"mode": "val"})
    for _ in range(num_iter):
        metric.before_iteration({"mode": "val"})
        metric.after_iteration({"mode": "val", "test": torch.tensor([1.1, 2.0])})
    metric.after_epoch({"mode": "val", "iter_per_epoch": num_iter})
    assert pytest.approx(metric.history["train"][0] == [1.0, 1.2])
    assert pytest.approx(metric.history["val"][0] == [1.1, 2.0])


loader = [(torch.randn(2, 10), torch.zeros(2, dtype=torch.long)) for _ in range(10)]
model = nn.Linear(10, 2)


def test_tqdm_reporters():
    c = TQDMReporter(range(4))
    with SupervisedTrainer(model, SGD(lr=0.1), nn.CrossEntropyLoss(),
                           callbacks=[AccuracyCallback(), c]) as t:
        for _ in c:
            t.train(loader)
            t.test(loader)


@pytest.mark.parametrize("c", [IOReporter("."), TensorboardReporter(".")])
def test_tb_reporters(c):
    with SupervisedTrainer(model, SGD(lr=0.1), nn.CrossEntropyLoss(),
                           callbacks=[AccuracyCallback(), c]) as t:
        for _ in range(10):
            t.train(loader)
            t.test(loader)
