""" Useful containers for PyTorch tensors and others
"""

import dataclasses
import types
from typing import Any, Type

import torch


class TensorTuple(tuple):
    """ Tuple for tensors.
    """

    def to(self, *args, **kwargs):
        """ Move stored tensors to a given device
        """

        return TensorTuple((t.to(*args, **kwargs) for t in self if isinstance(t, torch.Tensor)))


@dataclasses.dataclass
class TensorDataClass(object):
    """ TensorDataClass is an extension of `dataclass` that can handle tensors easily.
    """

    def __getitem__(self,
                    item: str
                    ) -> Any:
        # it is 50 times faster than using dataclasses.asdict
        return self.__dict__[item]

    def __iter__(self):
        # it is 50 times faster than using dataclasses.astuple
        return iter(self.__dict__.values())

    def to(self,
           *args,
           **kwargs):
        new = type(self)(*((t.to(*args, **kwargs) if isinstance(t, torch.Tensor) else t) for t in self))
        return new


def tensor_dataclass(cls=None,
                     **kwargs) -> TensorDataClass:
    """ Helper function to create a TensorDataClass, expected to be used as decorator::

        @tensor_dataclass
        class YourTensorClass(TensorDataClass):
            __slots__ = ('pred', 'loss')
            pred: torch.Tensor
            loss: torch.Tensor

        x = YourTensorClass(prediction, loss)
        x_cuda = x.to('cuda')
        x_int = x.to(dtype=torch.int32)
        registry_name, loss = x
        loss = x.loss
        loss = x['loss']

    :param cls: wrapped class
    :param kwargs: kwargs to dataclasses.dataclass
    :return:
    """

    def wrap(cls):
        # create cls whose baseclass is TensorDataClass
        cls = types.new_class(cls.__name__, (TensorDataClass,), {}, lambda ns: ns.update(cls.__dict__))
        # make cls to dataclass
        return dataclasses.dataclass(cls, **kwargs)

    return wrap if cls is None else wrap(cls)


class StepDict(dict):
    """ Dictionary with step, state_dict, load_state_dict and zero_grad. Intended to be used with Optimizer, lr_scheduler::

        sd = StepDict(Optimizer, generator=Adam(...), discriminator=Adam(...))
        sd.step()
        # is equivalent to generator_opt.step(); discriminator.step()

    :param _type:
    :param kwargs:
    """

    def __init__(self, _type: Type, **kwargs):
        super(StepDict, self).__init__(**kwargs)
        for k, v in self.items():
            if not (v is None or isinstance(v, _type)):
                raise RuntimeError(f"Expected {_type} as values but got {type(v)} with key ({k})")

    def step(self):
        for v in self.values():
            if hasattr(v, 'step'):
                v.step()

    def state_dict(self
                   ) -> dict[str, Any]:
        return {k: v.state_dict() for k, v in self.items()
                if hasattr(v, "state_dict")}

    def load_state_dict(self, state_dicts: dict):
        for k, v in state_dicts.items():
            if isinstance(v, dict):
                self[k].load_state_dict(v)

    def zero_grad(self):
        for v in self.values():
            if hasattr(v, "zero_grad"):
                v.zero_grad()
