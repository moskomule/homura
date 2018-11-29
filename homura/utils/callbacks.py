from collections import ChainMap
from pathlib import Path
from typing import Iterable, Callable

import torch

from ._vocabulary import *

__all__ = ["Callback", "MetricCallback", "CallbackList", "AccuracyCallback",
           "LossCallback", "WeightSave"]


class Callback(object):
    """
    Base class of Callback class
    """

    def before_iteration(self, data: dict) -> dict:
        pass

    def after_iteration(self, data: dict) -> dict:
        pass

    def before_epoch(self, data: dict) -> dict:
        pass

    def after_epoch(self, data: dict) -> dict:
        pass

    def before_all(self, data: dict) -> dict:
        pass

    def after_all(self, data: dict) -> dict:
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class MetricCallback(Callback):
    def __init__(self, metric: Callable[[dict], float], name: str):
        """
        Base class of MetricCallback class such as AccuracyCallback
        :param metric: metric function: (data) -> float
        :param name: name of the metric
        """
        if metric is not None:
            self.metric_function = metric
        self.metric_name = name
        self._last_iter = {}
        self._last_epoch = {}
        self._metrics_history = {}

    def before_iteration(self, data: dict):
        self._last_iter.clear()

    def after_iteration(self, data: dict):
        mode = data[MODE]
        key = self._get_key_name(mode)
        # if once this method is called after every iteration, self._last_iter is not None
        if self._last_iter.get(key) is None:
            metric = self.metric_function(data)
            self._last_iter[key] = metric
            self._metrics_history[key][-1] += metric
        return self._last_iter

    def before_epoch(self, data: dict):
        # initialization
        self._last_epoch.clear()
        mode = data[MODE]
        key = self._get_key_name(mode)
        if self._metrics_history.get(key) is None:
            self._metrics_history[key] = []
        else:
            self._metrics_history[key].append(0)

    def after_epoch(self, data: dict):
        mode = data[MODE]
        iter_per_epoch = data[ITER_PER_EPOCH]
        key = self._get_key_name(mode)
        # if once this method is called after every epoch, self._last_epoch is not None
        if self._last_epoch.get(key) is None:
            self._metrics_history[key][-1] /= iter_per_epoch
            self._last_epoch[key] = self._metrics_history[key][-1]
        return self._last_epoch

    def _get_key_name(self, name):
        return f"{self.metric_name}_{name}"


class CallbackList(Callback):
    def __init__(self, *callbacks: Iterable[Callback] or Callback):
        """
        collect some callbacks
        :param callbacks: callbacks
        """
        if callbacks is None:
            raise TypeError("callbacks is expected to be Callback but None")

        if not isinstance(callbacks, Iterable):
            callbacks = [callbacks]

        for c in callbacks:
            if not isinstance(c, Callback):
                raise TypeError(f"{c} is not a callback!")
        self._callbacks: Iterable[Callback] = list(callbacks)

    def before_iteration(self, data: dict):
        return self._cat([c.before_iteration(data) for c in self._callbacks])

    def after_iteration(self, data: dict):
        return self._cat([c.after_iteration(data) for c in self._callbacks])

    def before_epoch(self, data: dict):
        return self._cat([c.before_epoch(data) for c in self._callbacks])

    def after_epoch(self, data: dict):
        return self._cat([c.after_epoch(data) for c in self._callbacks])

    def after_all(self, data: dict):
        return self._cat([c.after_all(data) for c in self._callbacks])

    def close(self):
        for c in self._callbacks:
            c.close()

    @staticmethod
    def _cat(maps: list):
        # make callbacks' return to a single map
        maps = [m for m in maps if m is not None]
        return dict(ChainMap(*maps))


class AccuracyCallback(MetricCallback):
    def __init__(self, k: int = 1):
        """
        calculate and accumulate accuracy
        """
        self.top_k = k
        suffix = f"_top{self.top_k}" if self.top_k != 1 else ""
        super(AccuracyCallback, self).__init__(metric=self.accuracy, name=f"accuracy{suffix}")

    def accuracy(self, data):
        output, target = data[OUTPUT], data[INPUTS][1]
        with torch.autograd.no_grad():
            _, pred_idx = output.topk(self.top_k, dim=1)
            target = target.view(-1, 1).expand_as(pred_idx)
            return (pred_idx == target).float().sum(dim=1).mean().item()


class LossCallback(MetricCallback):
    def __init__(self):
        """
        accumulate loss
        """
        super(LossCallback, self).__init__(metric=lambda data: data[LOSS],
                                           name="loss")


class WeightSave(Callback):
    def __init__(self, save_path: str or Path, save_freq: int = 1):
        """
        save weights after every epoch
        :param save_path: path to be saved
        :param save_freq: frequency of saving in epoch
        """

        self.save_path = Path(save_path) / NOW
        self.save_freq = save_freq

        if not self.save_path.exists():
            self.save_path.mkdir(parents=True)

    def after_epoch(self, data: dict):
        if data[EPOCH] % self.save_freq == 0:
            try:
                torch.save({MODEL: data[MODEL].state_dict(),
                            OPTIMIZER: data[OPTIMIZER].state_dict(),
                            EPOCH: data[EPOCH]},
                           self.save_path / f"{data[EPOCH]}.pkl")
            except Exception as e:
                raise e
