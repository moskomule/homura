import dataclasses
from typing import Dict, Any, Type

import torch


class TensorTuple(tuple):
    """ Tuple for tensors.
    """

    def to(self, *args, **kwargs):
        """ Move stored tensors to a given device
        """

        return TensorTuple((t.to(*args, **kwargs) for t in self if torch.is_tensor(t)))


@dataclasses.dataclass
class TensorDataClass(object):
    """ TensorDataClass is an extension of `dataclass` that can handle tensors easily.
    It can be used as `NamedTensorTuple` ::

        @dataclasses.dataclass
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

    """

    def __getitem__(self,
                    item: str
                    ) -> Any:
        return dataclasses.asdict(self)[item]

    def __iter__(self):
        return iter((dataclasses.astuple(self)))

    def to(self,
           *args,
           **kwargs):
        new = type(self)(*((t.to(*args, **kwargs) if torch.is_tensor(t) else t) for t in self))
        return new


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
                   ) -> Dict[str, Any]:
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
