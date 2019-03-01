from abc import ABCMeta
from logging import Logger
from typing import Callable, Iterable, Dict, Optional

import torch
from torch import nn

from homura.callbacks import CallbackList, Callback
from homura.liblog import get_logger
from ._vocabulary import *


class Runner(metaclass=ABCMeta):
    """Meta-class for Trainer and Inferencer
    """

    def __init__(self, model: nn.Module or Dict[str, nn.Module],
                 callbacks: Optional[Callback or Iterable[Callable]] = None,
                 device: torch.device or str = None,
                 use_cudnn_benchmark=True, use_cuda_nonblocking=False, logger: Optional[Logger] = None, **kwargs):

        self.logger = get_logger(__name__) if logger is None else logger
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
            self.logger.debug(
                f"cuda: True, cudnn.benchmark: {use_cudnn_benchmark}, nonblocking: {use_cuda_nonblocking}")
        else:
            self._cuda_nonblocking = False
            self.logger.info("Running on CPU!")

        # set callback(s)
        if isinstance(callbacks, CallbackList):
            self._callbacks = callbacks
        elif isinstance(callbacks, Callback):
            self._callbacks = callbacks
            self.logger.debug(f"registered callback {callbacks.__class__.__name__}")
        elif isinstance(callbacks, Iterable):
            self._callbacks = CallbackList(*callbacks)
        elif callbacks is None:
            # if callback is not set
            self._callbacks = Callback()
            self.logger.debug(f"No callback registered")
        else:
            raise TypeError(f"type(callbacks) should not be {type(callbacks)}!")

        # set kwargs
        for k, v in kwargs.items():
            if hasattr(self, k):
                raise AttributeError(f"{self} already has {k}")
            if torch.is_tensor(v):
                v.to(self.device)
            setattr(self, k, v)
