from abc import ABCMeta
from collections import ChainMap
from collections.abc import Mapping
from pathlib import Path
from typing import Iterable, Callable, Any

import torch
from torch import distributed

from homura.liblog import get_logger
from .utils._vocabulary import *
from .utils.environment import is_distributed, get_global_rank
from .utils.miscs import get_git_hash

__all__ = ["Callback", "MetricCallback", "CallbackList", "AccuracyCallback",
           "LossCallback", "WeightSave", "metric_callback_decorator"]


class Callback(metaclass=ABCMeta):
    """ Base class of Callback class
    """

    def before_iteration(self, data: Mapping) -> Mapping:
        pass

    def after_iteration(self, data: Mapping) -> Mapping:
        pass

    def before_epoch(self, data: Mapping) -> Mapping:
        pass

    def after_epoch(self, data: Mapping) -> Mapping:
        pass

    def before_all(self, data: Mapping) -> Mapping:
        pass

    def after_all(self, data: Mapping) -> Mapping:
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class _NoOpCallback(Callback):
    def __init__(self, *args, **kwargs):
        pass


class MetricCallback(Callback):
    """ Base class of MetricCallback class such as AccuracyCallback

    :param metric: metric function: (data) -> float
    :param name: name of the metric
    """

    def __init__(self, metric: Callable[[Mapping], Any], name: str, logger=None):
        if metric is not None:
            self.metric_function = metric
        self.metric_name = name
        self._last_iter = {}
        self._last_epoch = {}
        self._metrics_history = {}
        self._logger = get_logger(__name__) if logger is None else logger
        self._warning_flag = True

    def before_iteration(self, data: Mapping):
        self._last_iter.clear()

    def after_iteration(self, data: Mapping):
        mode = data[MODE]
        key = self._get_key_name(mode)
        # To avoid calculate same metric multiple times.
        # If once this method is called after an iteration, self._last_iter is not None
        if self._last_iter.get(key) is None:
            metric = self.metric_function(data)
            if torch.is_tensor(metric):
                metric = self.reduce(metric)

            if metric is None:
                metric = 0
                if self._warning_flag:
                    self._logger.warning(f"{self.metric_function.__name__} get None and convert it to 0")
                    self._warning_flag = False

            self._last_iter[key] = metric

            if isinstance(metric, Mapping):
                if self._metrics_history[key][-1] == 0:
                    self._metrics_history[key][-1] = metric
                else:
                    self._metrics_history[key][-1] = {k: v + self.reduce(metric[k])
                                                      for k, v in self._metrics_history[key][-1].items()}
            else:
                self._metrics_history[key][-1] += metric
        return self._last_iter

    def before_epoch(self, data: Mapping):
        # initialization
        self._last_epoch.clear()
        mode = data[MODE]
        key = self._get_key_name(mode)
        if self._metrics_history.get(key) is None:
            self._metrics_history[key] = [0]
        else:
            self._metrics_history[key].append(0)

    def after_epoch(self, data: Mapping):
        mode = data[MODE]
        iter_per_epoch = data[ITER_PER_EPOCH]
        key = self._get_key_name(mode)
        # if once this method is called, self._last_epoch is not None
        if self._last_epoch.get(key) is None:
            if isinstance(self._metrics_history[key][-1], Mapping):
                self._metrics_history[key][-1] = {k: v / iter_per_epoch
                                                  for k, v in self._metrics_history[key][-1].items()}
            else:
                self._metrics_history[key][-1] /= iter_per_epoch
            self._last_epoch[key] = self._metrics_history[key][-1]
        return self._last_epoch

    def _get_key_name(self, name):
        return f"{self.metric_name}_{name}"

    @property
    def history(self) -> dict:
        """ History of metric.

        :return: dict of histories in {mode: [history]}

        Using this property, history of metrics can be used after a training loop ::

            >>> metric = ...
            >>> # training loop
            >>> final_result = metric.history["val"][-1]

        """

        return {k.split("_")[1]: v for k, v in self._metrics_history.items()}

    @staticmethod
    def reduce(tensor):
        """ for distributed setting
        """

        if not is_distributed or not torch.is_tensor(tensor):
            return tensor

        rt = tensor.cuda()
        distributed.all_reduce(rt, op=distributed.ReduceOp.SUM)
        return rt / distributed.get_world_size()


class CallbackList(Callback):
    """ Combine some callbacks

    :param callbacks: callbacks
    """

    def __init__(self, *callbacks: Iterable[Callback] or Callback):
        if callbacks is None:
            raise TypeError("callbacks is expected to be Callback but None")

        if not isinstance(callbacks, Iterable):
            callbacks = [callbacks]

        callbacks = [c for c in callbacks if c is not None]
        for c in callbacks:
            if not isinstance(c, Callback):
                raise TypeError(f"{c} is not callback!")
        self._callbacks: Iterable[Callback] = list(callbacks)

    def before_iteration(self, data: Mapping):
        return self._cat([c.before_iteration(data) for c in self._callbacks])

    def after_iteration(self, data: Mapping):
        return self._cat([c.after_iteration(data) for c in self._callbacks])

    def before_epoch(self, data: Mapping):
        return self._cat([c.before_epoch(data) for c in self._callbacks])

    def after_epoch(self, data: Mapping):
        return self._cat([c.after_epoch(data) for c in self._callbacks])

    def after_all(self, data: Mapping):
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
    """ Callback for accuracy

    :param k: report top-k accuracy
    """

    def __init__(self, k: int = 1):
        self.top_k = k
        suffix = f"_top{self.top_k}" if self.top_k != 1 else ""
        super(AccuracyCallback, self).__init__(metric=self.accuracy, name=f"accuracy{suffix}")

    def accuracy(self, data):
        output, target = data[OUTPUT], data[DATA][1]
        _, pred_idx = output.topk(self.top_k, dim=1)
        target = target.view(-1, 1).expand_as(pred_idx)
        return (pred_idx == target).float().sum(dim=1).mean()


class LossCallback(MetricCallback):
    """ Callback for loss function
    """

    def __init__(self):
        super(LossCallback, self).__init__(metric=lambda data: data[LOSS],
                                           name="loss")


def metric_callback_decorator(_metric: Callable = None, name: str = None):
    """ Decorator to create a metrics callback

        >>> @metric_callback_decorator("loss")
        >>> def loss(data):
        >>>     return data["loss"]
    """

    def wrapper(metric: Callable):
        return MetricCallback(metric, name=metric.__name__ if name is None else name)

    return wrapper if _metric is None else wrapper(_metric)


class WeightSave(Callback):
    """ Save weights after every epoch

    :param save_path: path to be saved
    :param save_freq: frequency of saving in epoch
    """

    def __new__(cls, *args, **kwargs):
        if get_global_rank() > 0:
            return _NoOpCallback
        else:
            return object.__new__(cls)

    def __init__(self, save_path: str or Path, save_freq: int = 1):

        self.save_path = Path(save_path) / (NOW + "-" + get_git_hash())
        self.save_freq = save_freq

        if not self.save_path.exists():
            self.save_path.mkdir(parents=True)

    def after_epoch(self, data: Mapping):
        if data[EPOCH] % self.save_freq == 0:
            try:
                torch.save({MODEL: data[MODEL].state_dict(),
                            OPTIMIZER: data[OPTIMIZER].state_dict(),
                            EPOCH: data[EPOCH],
                            STEP: data[STEP]},
                           self.save_path / f"{data[EPOCH]}.pkl")
            except Exception as e:
                raise e
