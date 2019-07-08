from collections.abc import MutableMapping
from copy import deepcopy

import torch

__all__ = ["Map", "TensorTuple", "StepDict"]


class Map(MutableMapping, dict):
    """ dict like object but: stored values can be subscribed and attributed.

        >>> m = Map(test="test")
        >>> m.test is m["test"]
    """

    # inherit `dict` to avoid problem with `backward_hook`s
    __default_methods = ["update", "keys", "items", "values", "clear",
                         "copy", "get", "pop", "to", "deepcopy"] + __import__('keyword').kwlist
    __slots__ = ["_data"]

    def __init__(self, **kwargs):
        super(Map, self).__init__()
        self._data = {}
        if len(kwargs) > 0:
            self._data.update(kwargs)

    def __getattr__(self, item):
        if item in self.__default_methods:
            return getattr(self, item)
        return self._data.get(item)

    def __setattr__(self, key, value):
        if key == "_data":
            # initialization!
            super(Map, self).__setattr__(key, value)
        elif key not in self.__default_methods:
            self._data[key] = value
        else:
            raise KeyError(f"{key} is a method name.")

    def __getitem__(self, item):
        return self.__getattr__(item)

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def __len__(self):
        return len(self._data)

    def __delitem__(self, key):
        del self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __repr__(self):
        _str = self.__class__.__name__ + "("
        for k, v in self._data.items():
            _str += f"{k}={str(v)}, "
        # to strip the las ", "
        return _str.strip(", ") + ")"

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def to(self, device: str, **kwargs):
        """ Move stored tensors to a given device
        """
        for k, v in self._data.items():
            if isinstance(v, torch.Tensor):
                self._data[k] = v.to(device, **kwargs)
        return self

    def deepcopy(self):
        new = Map()
        for k, v in self._data.items():
            if isinstance(v, torch.Tensor):
                # Only leave tensors support __deepcopy__
                # detach creates a new tensor
                new[k] = v.detach()
            else:
                new[k] = deepcopy(v)
        return new

    def copy(self):
        new = Map()
        new._data = self._data.copy()
        return new


class TensorTuple(tuple):
    """ Tuple for tensors.
    """

    def to(self, *args, **kwargs):
        return TensorTuple((t.to(*args, **kwargs) for t in self if torch.is_tensor(t)))


class StepDict(dict):
    """ Dictionary with step, state_dict, load_state_dict. Intended to be used with Optimizer, lr_scheduler

    :param _type:
    :param kwargs:
    """

    def __init__(self, _type, **kwargs):
        super(StepDict, self).__init__(**kwargs)
        for k, v in self.items():
            if not (v is None or isinstance(v, _type)):
                raise RuntimeError(f"Expected {_type} as values but got {type(v)} with key ({k})")

    def step(self):
        for v in self.values():
            if v is not None:
                v.step()

    def state_dict(self):
        d = {}
        for k, v in self.items():
            if hasattr(v, "state_dict"):
                d[k] = v.state_dict()
        return d

    def load_state_dict(self, state_dicts: dict):
        for k, v in state_dicts.items():
            if isinstance(v, dict):
                self[k].load_state_dict(v)

    def zero_grad(self):
        for v in self.values():
            if hasattr(v, "zero_grad"):
                v.zero_grad()
