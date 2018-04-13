from typing import Iterable
from .miscs import to_tensor


class Callback(object):
    def end_iteration(self, data):
        raise NotImplementedError

    def end_epoch(self, data):
        raise NotImplementedError

    def end_all(self, data):
        raise NotImplementedError


class CallbackList(Callback):
    def __init__(self, *callbacks):
        if not isinstance(callbacks, Iterable):
            callbacks = [callbacks]
        for c in callbacks:
            if not isinstance(c, Callback):
                raise TypeError(f"{c} is not a callback!")
        self._callbacks = callbacks

    def end_iteration(self, data):
        return [c.end_iteration(data) for c in self._callbacks]

    def end_epoch(self, data):
        return [c.end_epoch(data) for c in self._callbacks]

    def end_all(self, data):
        return [c.end_all(data) for c in self._callbacks]


class AccuracyCallback(Callback):
    def __init__(self, k=1, correct=None):
        if not isinstance(k, Iterable):
            k = [k]
        self._k = k
        if correct is not None:
            self.correct = correct

    def _shared(self, data: dict):
        return [self.correct(data["output"], data["target"], k)
                for k in self._k]

    def end_iteration(self, data: dict):
        return self._shared(data)

    def end_epoch(self, data):
        return self._shared(data)

    def end_all(self, data):
        return self._shared(data)

    @staticmethod
    def correct(input, target, k=1):
        input = to_tensor(input)
        target = to_tensor(target)

        _, pred_idx = input.topk(k, dim=1)
        target = target.view(-1, 1).expand_as(pred_idx)
        return (pred_idx == target).float().sum(dim=1).mean()
