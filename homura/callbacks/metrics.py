import warnings
from collections.abc import Mapping
from functools import partial
from typing import Callable, Any

import torch
from torch import distributed

from homura.liblog import get_logger
from homura.metrics import confusion_matrix
from .base import Callback
from ..utils._vocabulary import *
from ..utils.environment import is_distributed, is_horovod_available


class MetricCallback(Callback):
    """ Base class of MetricCallback class such as AccuracyCallback

    :param metric: metric function: (data) -> float
    :param name: registry_name of the metric
    :param logger:
    :param no_reduce: skip reducing when distributed
    :param reduction: reduction method after every epoch
    """

    def __init__(self,
                 metric: Callable[[Mapping], Any],
                 name: str,
                 logger=None,
                 reduction="average",
                 no_reduce: bool = False):

        if metric is not None:
            self.metric_function = metric
        self.metric_name = name
        self._last_iter = {}
        self._last_epoch = {}
        self._metrics_history = {}
        self._logger = get_logger(__name__) if logger is None else logger
        self._no_reduce = no_reduce

        if reduction not in ("average", "sum"):
            raise RuntimeError(f"`reduction` should be 'average' or 'sum', but got {reduction} instead")
        self.reduction = reduction

    def before_iteration(self,
                         data: Mapping):
        self._last_iter.clear()

    def after_iteration(self,
                        data: Mapping):
        mode = data[MODE]
        key = self._get_key_name(mode)
        # To avoid calculate same metric multiple times.
        # If once this method is called after an iteration, self._last_iter is not None
        if self._last_iter.get(key) is None:
            # Note that `metric` can be GPU tensor
            metric = self.metric_function(data)

            if metric is None:
                metric = 0
                # warnings.warn warns once
                warnings.warn(f"{self.metric_function.__name__} get None and convert it to 0")

            self._last_iter[key] = metric

            if isinstance(metric, Mapping):
                # first time
                if self._metrics_history[key][-1] is None:
                    self._metrics_history[key][-1] = {k: self.to_cpu(self.reduce(metric[k])) for k in metric.keys()}
                else:
                    self._metrics_history[key][-1] = {k: v + self.to_cpu(self.reduce(metric[k]))
                                                      for k, v in self._metrics_history[key][-1].items()}
            else:
                if self._metrics_history[key][-1] is None:
                    self._metrics_history[key][-1] = 0
                self._metrics_history[key][-1] += self.to_cpu(self.reduce(metric))
        return self._last_iter

    def before_epoch(self,
                     data: Mapping):
        mode = data[MODE]
        key = self._get_key_name(mode)

        # before_all
        if self._metrics_history.get(key) is None:
            self._metrics_history[key] = [None]

        # first time at each epoch
        if self._last_epoch != {}:
            # initialization for this epoch
            self._last_epoch.clear()
            if self._metrics_history[key][-1] is not None:
                self._metrics_history[key].append(None)

    def after_epoch(self,
                    data: Mapping):
        mode = data[MODE]
        divisor = data[ITER_PER_EPOCH] if self.reduction == "average" else 1
        key = self._get_key_name(mode)
        # if once this method is called, self._last_epoch[key] is not None
        if self._last_epoch.get(key) is None:
            if isinstance(self._metrics_history[key][-1], Mapping):
                self._metrics_history[key][-1] = {k: v / divisor
                                                  for k, v in self._metrics_history[key][-1].items()}
            else:
                self._metrics_history[key][-1] /= divisor
            self._last_epoch[key] = self._metrics_history[key][-1]
        return self._last_epoch

    def _get_key_name(self,
                      name: str):
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

        history = {}
        for k, v in self._metrics_history.items():
            name = "_".join(k.split("_")[1:])
            if torch.is_tensor(v[0]):
                history[name] = torch.tensor(v).tolist() if v[0].numel() == 1 else torch.stack(v).tolist()
            else:
                history[name] = v

        return history

    def reduce(self,
               tensor: torch.Tensor):

        if not torch.is_tensor(tensor):
            return tensor
        if is_distributed() and not self._no_reduce:
            if is_horovod_available():
                import horovod.torch as hvd

                # hvd's all_reduce applies average
                return hvd.allreduce(tensor)
            # pytorch's all_reduce does not applies average
            distributed.all_reduce(tensor, op=distributed.ReduceOp.SUM)
            return tensor / distributed.get_world_size()
        return tensor

    @staticmethod
    def to_cpu(tensor):
        # tensor can be torch.Tensor or Number
        if torch.is_tensor(tensor):
            return tensor.cpu()
        return tensor


class AccuracyCallback(MetricCallback):
    """ Callback for accuracy

    :param k: report top-k accuracy
    :param target_index: index of `target` in `data`
    """

    def __init__(self,
                 k: int = 1,
                 target_index: int = 1):
        self.top_k = k
        suffix = f"_top{self.top_k}" if self.top_k != 1 else ""
        self._accuracy = partial(torch.argmax, dim=1, keepdim=True) if k == 1 \
            else lambda x: partial(torch.topk, k=k, dim=1)(x)[1]
        self.target_index = target_index
        super(AccuracyCallback, self).__init__(metric=self.accuracy, name=f"accuracy{suffix}")

    def accuracy(self,
                 data: Mapping):
        output, target = data[OUTPUT], data[DATA][self.target_index]
        pred_idx = self._accuracy(output)
        target = target.view(-1, 1).expand_as(pred_idx)
        return (pred_idx == target).float().sum(dim=1).mean()


class LossCallback(MetricCallback):
    """ Callback for loss function
    """

    def __init__(self):
        super(LossCallback, self).__init__(metric=lambda data: data[LOSS],
                                           name="loss")


def metric_callback_decorator(_metric: Callable = None,
                              *,
                              name: str = None,
                              reduction="average"):
    """ Decorator to create a metric callback

        >>> @metric_callback_decorator("loss")
        >>> def loss(data):
        >>>     return data["loss"]
    """

    def wrapper(metric: Callable):
        return MetricCallback(metric,
                              name=metric.__name__ if name is None else name,
                              reduction=reduction)

    return wrapper if _metric is None else wrapper(_metric)


def metric_callback_by_name(name: str):
    """ Construct a metric callback by ``registry_name`` (key) of ``data``.

    :param name:
    :return:
    """

    @metric_callback_decorator(name=name)
    def f(data):
        return data[name]

    return f


class IOUCallback(MetricCallback):
    """ Callback for IOU (classwise and mean IOU)
    """

    def __init__(self):
        super(IOUCallback, self).__init__(self.cm,
                                          "iou",
                                          reduction="sum")

    def cm(self, data):
        output, target = data["output"], data["data"][1]
        return confusion_matrix(output, target)

    def after_epoch(self, data):
        cm_map = super(IOUCallback, self).after_epoch(data)
        name, cm = tuple(cm_map.items())[0]
        _, post = name.split("_")
        cm = cm.float()
        iou = cm.diag() / (cm.sum(0) + cm.sum(1) - cm.diag())
        return {f"iou_{post}": iou,
                f"miou_{post}": iou.mean()}
