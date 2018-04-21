from typing import Iterable, Callable
from collections import ChainMap
from pathlib import Path

import torch

from ._vocabulary import V
from .reporter import Reporter

__all__ = ["Callback", "MetricCallback", "CallbackList", "AccuracyCallback",
           "LossCallback", "WeightSave", "ReporterCallback"]


class Callback(object):
    """
    Base class of Callback class
    """

    def start_iteration(self, data: dict) -> dict:
        pass

    def end_iteration(self, data: dict) -> dict:
        pass

    def start_epoch(self, data: dict) -> dict:
        pass

    def end_epoch(self, data: dict) -> dict:
        pass

    def end_all(self, data: dict) -> dict:
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class MetricCallback(Callback):
    def __init__(self, metric: Callable[[dict, int], float], top_k: int or tuple, name: str):
        """
        Base class of MetricCallback class such as AccuracyCallback
        :param metric: metric function: (data, k) -> float, where k is from top_k
        :param top_k: top Nth metric (e.g. (1, 5) then top 1 and 5 accuracy)
        :param name: name of the metric
        """
        if not isinstance(top_k, Iterable):
            top_k = [top_k]
        self.top_k = top_k
        if metric is not None:
            self.metric_function = metric
        self.metric_name = name
        self._metrics_history = {}

    def end_iteration(self, data: dict):
        _iter_metrics = {}
        name = data[V.NAME]
        for k in self.top_k:
            key = self._key(k, name)
            metric = self.metric_function(data, k)
            _iter_metrics[key] = metric
            if self._metrics_history.get(key) is None:
                # initialize
                self._metrics_history[key] = [metric]
            else:
                self._metrics_history[key][-1] += metric
        return _iter_metrics

    def start_epoch(self, data: dict):
        name = data[V.NAME]
        for k in self.top_k:
            key = self._key(k, name)
            self._metrics_history[key].append(0)

    def end_epoch(self, data: dict):
        _epoch_metrics = {}
        name = data[V.NAME]
        iter_per_epoch = data[V.ITER_PER_EPOCH]
        for k in self.top_k:
            key = self._key(k, name)
            self._metrics_history[key][-1] /= iter_per_epoch
            _epoch_metrics[key] = self._metrics_history[key][-1]

        return _epoch_metrics

    def _key(self, k, name):
        if len(self.top_k) == 1:
            return f"{self.metric_name}_{name}"
        else:
            return f"{self.metric_name}_{name}_top{k}"


class CallbackList(Callback):
    def __init__(self, *callbacks: Callback):
        """
        collect some callbacks
        :param callbacks: callbacks
        """
        if callbacks is None:
            callbacks = Callback()

        if not isinstance(callbacks, Iterable):
            callbacks = [callbacks]

        for c in callbacks:
            if not isinstance(c, Callback):
                raise TypeError(f"{c} is not a callback!")
        self._callbacks = callbacks

    def start_iteration(self, data: dict):
        return self._cat([c.start_iteration(data) for c in self._callbacks])

    def end_iteration(self, data: dict):
        return self._cat([c.end_iteration(data) for c in self._callbacks])

    def start_epoch(self, data: dict):
        return self._cat([c.start_epoch(data) for c in self._callbacks])

    def end_epoch(self, data: dict):
        return self._cat([c.end_epoch(data) for c in self._callbacks])

    def end_all(self, data: dict):
        return self._cat([c.end_all(data) for c in self._callbacks])

    def close(self):
        for c in self._callbacks:
            c.close()

    @staticmethod
    def _cat(maps: list):
        maps = [m for m in maps if m is not None]
        return dict(ChainMap(*maps))


class AccuracyCallback(MetricCallback):
    def __init__(self, k: int or tuple = 1):
        """
        calculate and accumulate accuracy
        :param k: top k accuracy (e.g. (1, 5) then top 1 and 5 accuracy)
        """
        super(AccuracyCallback, self).__init__(metric=self.accuracy, top_k=k, name="accuracy")

    @staticmethod
    def accuracy(data, k=1):
        input, target = data[V.OUTPUT], data[V.TARGET]
        with torch.autograd.no_grad():

            _, pred_idx = input.topk(k, dim=1)
            target = target.view(-1, 1).expand_as(pred_idx)
            return (pred_idx == target).float().sum(dim=1).mean()


class LossCallback(MetricCallback):
    def __init__(self):
        """
        accumulate loss
        """
        super(LossCallback, self).__init__(metric=lambda data, _: data[V.LOSS],
                                           top_k=1, name="loss")


class WeightSave(Callback):
    def __init__(self, save_path: str or Path, save_freq: int = 1):
        """
        save weights after every epoch
        :param save_path: path to be saved
        :param save_freq: frequency of saving
        """

        self.save_path = Path(save_path) / V.NOW
        self.save_freq = save_freq

        if not self.save_path.exists():
            self.save_path.mkdir(parents=True)

    def end_epoch(self, data: dict):
        if data[V.EPOCH] % self.save_freq:
            torch.save({V.MODEL: data[V.MODEL].state_dict(),
                        V.OPTIMIZER: data[V.OPTIMIZER].state_dict(),
                        V.EPOCH: data[V.EPOCH]},
                       self.save_path / f"{data[V.EPOCH]}.pkl")


class ReporterCallback(Callback):
    def __init__(self, reporter: Reporter, callback: Callback, *,
                 report_freq: int = -1):
        """
        reporter integrated callback
        :param reporter:
        :param callback:
        :param report_freq: report frequency in step. If -1, no report during each iteration.
        """
        self.reporter = reporter
        self.callback = callback
        self.report_freq = report_freq

    def end_iteration(self, data: dict):
        results = self.callback.end_iteration(data)

        if (data[V.STEP] % self.report_freq == 0) and self.report_freq > 0:
            for k, v in results.items():
                self.reporter.add_scalar(v, name=f"{V.STEP}_{k}", idx=data[V.STEP])

    def end_epoch(self, data: dict):
        results = self.callback.end_epoch(data)
        for k, v in results.items():
            self.reporter.add_scalar(v, name=k, idx=data[V.EPOCH])

    def close(self):
        self.reporter.close()
        self.callback.close()
