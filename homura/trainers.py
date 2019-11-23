from abc import ABCMeta, abstractmethod
from functools import partial as Partial
from types import MethodType
from typing import Callable, Iterable, Dict, Mapping, Tuple, Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as Scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from homura import is_distributed
from homura.liblog import get_logger
from .callbacks import Callback, CallbackList, WeightSave
from .callbacks.reporters import _ReporterBase
from .utils._vocabulary import *
from .utils.containers import TensorTuple, Map, StepDict
from .utils.environment import get_global_rank, get_local_rank, init_distributed

__all__ = ["TrainerBase", "SupervisedTrainer"]


class TrainerBase(metaclass=ABCMeta):
    """ Train and evaluate model in a given period (an epoch or iterations)

    :param model: model to be trained in `nn.Module` or `{"name": nn.Module}`
    :param optimizer: optimizer for the model in `partial`, `torch.optim.Optimizer` or dict of them. For distributed
    training, optimizer like `partial(SGD)` is recommended. See `homura.optim`.
    :param loss_f: loss function
    :param callbacks: callbacks
    :param scheduler: scheduler for the model in `partial`, `lr_scheduler._LRScheduler` or dict of them
    :param update_scheduler_by_epoch: `True` if scheduler is updated by epoch, otherwise update by iteration
    :param device:
    :param verb:
    :param use_cudnn_benchmark:
    :param use_cuda_nonblocking:
    :param logger:
    :param distributed_backend:
    :param init_method:
    :param use_sync_bn:
    :param tqdm_ncols:
    :param kwargs:
    """

    def __init__(self,
                 model: nn.Module or Dict[str, nn.Module],
                 optimizer: Optional[Partial or Optimizer or Dict[str, Optimizer]],
                 loss_f: Optional[Callable or Dict[str, Callable]],
                 *,
                 callbacks: Optional[Iterable[Callback]] = None,
                 scheduler: Optional[Partial or Scheduler or Dict[str, Scheduler]] = None,
                 update_scheduler_by_epoch: bool = True,
                 device: Optional[torch.device or str] = None,
                 verb=True,
                 use_cudnn_benchmark=True,
                 use_cuda_nonblocking=False,
                 logger=None,
                 distributed_backend="nccl",
                 init_method="env://",
                 use_sync_bn: bool = False,
                 tqdm_ncols: int = 80,
                 **kwargs):

        if logger is None:
            logger = get_logger(__name__)
        self.logger = logger

        if device is None:
            self.device = torch.device(GPU) if torch.cuda.is_available() else torch.device(CPU)
        else:
            self.device = device

        if is_distributed():
            if use_sync_bn:
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

            # just in case `init_distributed` is not called yet
            init_distributed(distributed_backend, init_method, warning=False)
            rank = get_local_rank()
            torch.cuda.set_device(rank)
            self.device = torch.device(f"{GPU}:{rank}")
            if get_global_rank() > 0:
                # to avoid overwriting
                verb = False

        if isinstance(model, nn.Module):
            self.model = model
        elif isinstance(model, dict):
            self.model = nn.ModuleDict(model)
        else:
            raise TypeError(f"Unknown type for `model`. Expected nn.Module or Dict[str, Module] but got {type(model)}")

        if GPU in str(self.device):
            self.model.to(self.device)
            torch.backends.cudnn.benchmark = use_cudnn_benchmark
            self._cuda_nonblocking = use_cuda_nonblocking
            self.logger.debug(f"cuda: True, cudnn.benchmark: {use_cudnn_benchmark}, "
                              f"cuda.nonblocking: {use_cuda_nonblocking}")
        else:
            self._cuda_nonblocking = False
            # usually, this is not expected
            self.logger.info(f"cuda: False (torch.cuda.is_available()={torch.cuda.is_available()})")

        if is_distributed():
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[rank])

        if isinstance(self.model, nn.parallel.DistributedDataParallel) or isinstance(self.model, nn.DataParallel):
            self.accessible_model = self.model.module
        else:
            self.accessible_model = self.model

        self.optimizer = None
        self.scheduler = None
        self._callbacks = None
        self.update_scheduler_by_epoch = update_scheduler_by_epoch
        self._set_optimizer(optimizer)
        self._set_scheduler(scheduler)
        self._set_callbacks(callbacks)

        self.loss_f = loss_f
        self._verb = verb

        # called via property
        # _step and _epoch are set to -1 because they are incremented before each iteration and epoch!
        self._step = -1
        self._epoch = -1
        self._is_train = True
        self._tqdm = Partial(tqdm, ncols=tqdm_ncols) if verb else lambda x: x

        _map_base = {MODEL: self.accessible_model,
                     OPTIMIZER: self.optimizer,
                     SCHEDULER: self.scheduler,
                     TRAINER: self}
        self._iteration_map = Map(**_map_base.copy())
        self._epoch_map = Map(**_map_base.copy())
        self._all_map = Map(**_map_base.copy())

        for k, v in kwargs.items():
            if hasattr(self, k):
                raise AttributeError(f"{self} already has {k}")
            if torch.is_tensor(v):
                v = v.to(self.device)
            if isinstance(v, nn.Module):
                v.to(self.device)
            setattr(self, k, v)

        self._callbacks.before_all(self._all_map)

    @property
    def step(self):
        return self._step

    @property
    def epoch(self):
        return self._epoch

    @property
    def is_train(self):
        return self._is_train

    @abstractmethod
    def iteration(self,
                  data: Tuple[torch.Tensor]) -> Mapping[str, torch.Tensor]:
        """ Iteration part, user can override via duck typing or override_iteration ::

            def iteration(self, data: Tuple[torch.Tensor]) -> Mapping[str, torch.Tensor]:
                input, target = data
                output = self.model(input)
                loss = self.loss_f(output, target)
                if self.is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                return Map(loss=loss, output=output)

        :param data: data used during a iteration
        :return: loss, output
        """

    def override_iteration(self,
                           new_iteration: Callable):
        """ Override iteration method ::

            def new_iteration(trainer, data):
                input, target = data
                ...
                results.loss = loss
                return results
            trainer.update_iteration(new_iteration)

        :param new_iteration:
        :return:
        """
        setattr(self, "iteration", MethodType(new_iteration, self))
        self.logger.debug("Override iteration")

    def _iteration(self,
                   data: Tuple[torch.Tensor],
                   mode: str):
        """ Iteration level training loop

        :param data: should be TensorTuple
        :param mode: train, test or val
        :return:
        """

        self._iteration_map.update({EPOCH: self.epoch, ITERATION: self.step, MODE: mode})
        with torch.no_grad():
            self._callbacks.before_iteration(self._iteration_map)
        results = self.iteration(data)
        if self.is_train and self.scheduler is not None and not self.update_scheduler_by_epoch:
            self.scheduler.step()
        # backward compatibility
        if isinstance(results, tuple):
            loss, output = TensorTuple(results)
            results = dict(loss=loss, output=output)
            self._iteration_map.update(**results)
        else:
            self._iteration_map.update(**results)
        self._iteration_map[DATA] = data
        with torch.no_grad():
            self._callbacks.after_iteration(self._iteration_map)
        # clean up
        self._iteration_map.pop(DATA)
        for k in results.keys():
            self._iteration_map.pop(k)

    def __enter__(self):
        """

        >>> with TrainerBase(...) as trainer:
        >>>     trainer.train(...)

        """

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit()

    def _loop(self,
              data_loader: Iterable or DataLoader,
              mode: str):

        self._epoch_map.update({EPOCH: self.epoch,
                                ITERATION: self.step,
                                MODE: mode,
                                ITER_PER_EPOCH: len(data_loader)})
        with torch.no_grad():
            self._callbacks.before_epoch(self._epoch_map)

        for data in self._tqdm(data_loader):
            data = TensorTuple(data).to(self.device, non_blocking=self._cuda_nonblocking)
            if self.is_train:
                # increment step here for `callbacks`
                self._step += 1
            self._iteration(data, mode)

        with torch.no_grad():
            self._callbacks.after_epoch(self._epoch_map)
        self.logger.debug(f"epoch {self.epoch} finished")

    def train(self,
              data_loader: Iterable or DataLoader,
              mode: str = TRAIN):
        """ Training the model for an epoch.

        :param data_loader:
        :param mode: Name of this loop. Default is `train`. Passed to callbacks.
        """

        self._is_train = True
        self._epoch += 1
        self.model.train()
        if hasattr(self.loss_f, "train"):
            self.loss_f.train()
        with torch.enable_grad():
            self._loop(data_loader, mode=mode)

        if self.scheduler is not None and self.update_scheduler_by_epoch:
            self.scheduler.step()

    def test(self,
             data_loader: Iterable or DataLoader,
             mode: str = TEST):
        """ Evaluate the model.

        :param data_loader:
        :param mode: Name of this loop. Default is `test`. Passed to callbacks.
        :return:
        """

        self._is_train = False
        self.model.eval()
        if hasattr(self.loss_f, "eval"):
            self.loss_f.eval()
        with torch.no_grad():
            self._loop(data_loader, mode=mode)

    def run(self,
            train_loader: Iterable or DataLoader,
            val_loader: Iterable or DataLoader,
            total_iterations: int,
            val_intervals: int):
        """ Train the model for a given iterations

        :param train_loader:
        :param val_loader:
        :param total_iterations:
        :param val_intervals:
        :return:
        """

        class ProxyLoader(object):
            def __init__(self, loader):
                self.loader = loader

            def __len__(self):
                return val_intervals

            def __iter__(self):
                counter = 0
                while True:
                    for data in self.loader:
                        if counter == val_intervals:
                            return  # from python 3.7, this is valid
                        yield data
                        counter += 1

        train_loader = ProxyLoader(train_loader)

        for ep in range(total_iterations // val_intervals):
            self.train(train_loader)
            self.test(val_loader)

    def exit(self):
        with torch.no_grad():
            self._all_map.update({EPOCH: self.epoch, ITERATION: self.step})
            self._callbacks.after_all(self._all_map)
            self._callbacks.close()

    def _set_optimizer(self,
                       optimizer: Optional[Partial or Optimizer or Dict[str, Optimizer]]):
        if isinstance(optimizer, Optimizer) or optimizer is None:
            self.optimizer = optimizer

        elif isinstance(optimizer, Partial):
            if not issubclass(optimizer.func, Optimizer):
                raise TypeError(f"`optimizer.func` is expected to be subclass of `Optimizer`"
                                f" but got {type(optimizer.func)}")
            self.optimizer = optimizer(self.model.parameters())

        elif isinstance(optimizer, dict):
            if not isinstance(self.model, nn.ModuleDict):
                raise TypeError("When `optimizer` is `dict`, `model` also needs to be "
                                "`dict` or `nn.ModuleDict`")
            self.optimizer = StepDict(Optimizer, **optimizer)

        else:
            raise TypeError(f"Unexpected type {type(optimizer)} for `optimizer`")

    def _set_scheduler(self,
                       scheduler: Optional[Partial or Scheduler or Dict[str, Scheduler]]):
        if self.optimizer is None:
            raise TypeError("Optimizer is not set, so scheduler cannot be set")

        if isinstance(scheduler, Scheduler) or scheduler is None:
            self.scheduler = scheduler

        elif isinstance(scheduler, Partial):
            if not issubclass(scheduler.func, Scheduler):
                raise TypeError(f"`scheduler.func` is expected to be subclass of `_LRScheduler`"
                                f" but got {type(scheduler.func)}")
            self.scheduler = scheduler(self.optimizer)

        elif isinstance(scheduler, dict):
            if not isinstance(self.optimizer, StepDict):
                raise TypeError("When `scheduler` is `dict`, `optimizer` is also needs to be `dict`")
            self.scheduler = StepDict(Scheduler, **scheduler)

        else:
            raise TypeError(f"Unexpected type {type(scheduler)} for `scheduler`")

    def _set_callbacks(self,
                       callbacks: Optional[Iterable[Callback]]):

        if callbacks is None:
            self._callbacks = Callback()
            return

        _reporters = []
        _outers = []
        _inners = []

        for c in callbacks:
            if isinstance(c, _ReporterBase):
                _reporters.append(c)
            elif isinstance(c, WeightSave):
                _outers.append(c)
            elif isinstance(c, Callback):
                _inners.append(c)
            else:
                raise TypeError(f"Element of `callbacks` is expected to be `Callback`, but got `{type(c)}`")

        _inners = CallbackList(_inners)
        for r in _reporters:
            r.register_callbacks(_inners)
        self._callbacks = CallbackList(_reporters + _outers)


class SupervisedTrainer(TrainerBase):
    def __init__(self,
                 model: nn.Module,
                 optimizer: Optimizer,
                 loss_f: Callable,
                 *,
                 callbacks: Optional[Callback or Iterable[Callable]] = None,
                 scheduler: Optional[Scheduler] = None,
                 verb=True,
                 use_cudnn_benchmark=True,
                 data_parallel=False,
                 **kwargs):
        if isinstance(model, dict):
            raise TypeError(f"{type(self)} does not support dict model")
        super(SupervisedTrainer, self).__init__(model, optimizer, loss_f, callbacks=callbacks, scheduler=scheduler,
                                                verb=verb, use_cudnn_benchmark=use_cudnn_benchmark, **kwargs)
        if data_parallel and not isinstance(self.model, nn.DataParallel) and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            self.model.to(self.device)

    def iteration(self,
                  data: Tuple[torch.Tensor, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        input, target = data
        output = self.model(input)
        loss = self.loss_f(output, target)
        if self.is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return Map(loss=loss, output=output)
