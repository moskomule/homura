from abc import ABCMeta
from typing import Callable, Iterable, Dict

import torch
from torch import nn

from .reporter.callbacks import CallbackList, Callback
from ._vocabulary import *


class Runner(metaclass=ABCMeta):

    def __init__(self, model: nn.Module or Dict[str, nn.Module],
                 callbacks: Callback or Iterable[Callable] = None,
                 device: torch.device or str = None,
                 use_cudnn_benchmark=True, use_cuda_nonblocking=False, **kwargs):
        """Meta-class for Trainer and Inferencer
        """

        if device is None:
            self.device = GPU if torch.cuda.is_available() else CPU
        else:
            self.device = device

        # set model(s)
        if isinstance(model, nn.Module):
            self.model = model
            self._is_single_model = True
        elif isinstance(model, dict):
            self.model = nn.ModuleDict(model)
            self._is_single_model = False
        else:
            raise TypeError(f"Unknown type for arg. model. Expected nn.Module or "
                            f"Dict[str, Module] but got {type(model)}")

        if GPU in str(self.device):
            if use_cudnn_benchmark:
                torch.backends.cudnn.benchmark = True
            self.model.to(self.device)
            self._cuda_nonblocking = use_cuda_nonblocking

        # set callback(s)
        if isinstance(callbacks, CallbackList):
            self._callbacks = callbacks
        elif isinstance(callbacks, Callback):
            self._callbacks = callbacks
        elif isinstance(callbacks, Iterable):
            self._callbacks = CallbackList(*callbacks)
        elif callbacks is None:
            # if callback is not set
            self._callbacks = Callback()
        else:
            raise TypeError(f"type(callbacks) should not be {type(callbacks)}!")

        # set kwargs
        for k, v in kwargs.items():
            if hasattr(self, k):
                raise AttributeError(f"{self} already has {k}")
            setattr(self, k, v)
