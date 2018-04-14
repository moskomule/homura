from typing import Iterable, Callable
from .miscs import to_tensor
from .vocabulary import V


class Callback(object):
    """
    Base class of Callback class
    """

    def start_iteration(self, data: dict):
        pass

    def end_iteration(self, data: dict):
        pass

    def start_epoch(self, data: dict):
        pass

    def end_epoch(self, data: dict):
        pass

    def end_all(self, data: dict):
        pass

    def close(self):
        pass


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

    def end_epoch(self, data: dict):
        _epoch_metrics = {}
        name = data[V.NAME]
        iter_per_epoch = data[V.ITER_PER_EPOCH]
        for k in self.top_k:
            key = self._key(k, name)
            self._metrics_history[key][-1] /= iter_per_epoch
            _epoch_metrics[key] = self._metrics_history[key][-1]
            self._metrics_history[key].append(0)
        return _epoch_metrics

    def end_all(self, data: dict):
        # remove 0 which self.end_epoch adds at the final epoch
        return {k: v[:-1] for k, v in self._metrics_history.items()}

    def _key(self, k, name):
        if len(self.top_k) == 1:
            return f"{self.metric_name}_{name}"
        else:
            return f"{self.metric_name}_{name}_top{k}"


class CallbackList(Callback):
    def __init__(self, *callbacks: Callback):
        if callbacks is None:
            callbacks = Callback()

        if not isinstance(callbacks, Iterable):
            callbacks = [callbacks]

        for c in callbacks:
            if not isinstance(c, Callback):
                raise TypeError(f"{c} is not a callback!")
        self._callbacks = callbacks

    def start_iteration(self, data: dict):
        return [c.start_iteration(data) for c in self._callbacks]

    def end_iteration(self, data: dict):
        return [c.end_iteration(data) for c in self._callbacks]

    def start_epoch(self, data: dict):
        return [c.start_epoch(data) for c in self._callbacks]

    def end_epoch(self, data: dict):
        return [c.end_epoch(data) for c in self._callbacks]

    def end_all(self, data: dict):
        return [c.end_all(data) for c in self._callbacks]

    def close(self):
        for c in self._callbacks:
            c.close()


class AccuracyCallback(MetricCallback):
    def __init__(self, k: int or tuple = 1):
        super(AccuracyCallback, self).__init__(metric=self.accuracy, top_k=k, name="accuracy")

    @staticmethod
    def accuracy(data, k=1):
        input, target = data[V.OUTPUT], data[V.TARGET]
        input = to_tensor(input)
        target = to_tensor(target)

        _, pred_idx = input.topk(k, dim=1)
        target = target.view(-1, 1).expand_as(pred_idx)
        return (pred_idx == target).float().sum(dim=1).mean()


class LossCallback(MetricCallback):
    def __init__(self):
        super(LossCallback, self).__init__(metric=lambda data, _: data[V.LOSS],
                                           top_k=1, name="loss")
