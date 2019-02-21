import torch

from homura import callbacks


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

    assert metric.history["train"] == [1]
    assert metric.history["val"] == [0.1]


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
    assert all(metric.history["train"][0] == torch.tensor([1.0, 1.2]))
    assert all(metric.history["val"][0] == torch.tensor([1.1, 2.0]))
