from __future__ import annotations

import contextlib
import warnings
from abc import ABCMeta, abstractmethod
from functools import partial as Partial
from types import MethodType
from typing import Any, Callable, Iterable, TypeVar

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as Scheduler
from torch.utils.data import DataLoader

from homura import get_global_rank, get_local_rank, is_distributed, is_master
from homura.liblog import get_logger, set_tqdm_stdout_stderr, set_verb_level, tqdm
from .metrics import accuracy
from .optim import LARC
from .reporters import ReporterList, TQDMReporter, _ReporterBase
from .utils._mixin import StateDictMixIn
from .utils.containers import StepDict, TensorTuple

__all__ = ["TrainerBase", "SupervisedTrainer"]

DataType = TypeVar('DataType', Tensor, tuple[Tensor, ...])


class TrainerBase(StateDictMixIn, metaclass=ABCMeta):
    """ Baseclass for Trainers

    :param model: model to be trained
    :param optimizer: optimizer for the model
    :param loss_f: loss function for training
    :param reporters: list of reporters
    :param scheduler: learning rate scheduler
    :param device: device to be used
    :param quiet: True to disable tqdm
    :param disable_cudnn_benchmark: True to disable cudnn benchmark mode
    :param disable_cuda_nonblocking: True to disable cuda nonblocking
    :param logger: optional logger
    :param use_sync_bn: True to convert BN to sync BN
    :param tqdm_ncols: number of columns of tqdm
    :param kwargs:
    """

    def __init__(self,
                 model: nn.Module or dict[str, nn.Module],
                 optimizer: Partial or Optimizer or dict[str, Optimizer],
                 loss_f: Callable or dict[str, Callable] = None,
                 *,
                 reporters: _ReporterBase or list[_ReporterBase] = None,
                 scheduler: Partial or Scheduler or dict[str, Scheduler] = None,
                 device: torch.device or str = None,
                 quiet: bool = False,
                 disable_cudnn_benchmark: bool = False,
                 disable_cuda_nonblocking: bool = False,
                 logger=None,
                 use_sync_bn: bool = False,
                 tqdm_ncols: int = 120,
                 debug: bool = False,
                 profile: bool = False,
                 dist_kwargs: dict = None,
                 prof_kwargs: dict = None,
                 disable_auto_ddp: bool = False,
                 **kwargs):

        if kwargs.get("update_scheduler_by_epoch"):
            raise DeprecationWarning("update_scheduler_by_epoch is deprecated, users need to step")

        if kwargs.get("callbacks"):
            raise DeprecationWarning("callback is deprecated, if you need, use homura before v2020.8")

        self.logger = logger or get_logger(__name__)
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self._is_debug = debug

        if self._is_debug:
            self.logger.warning("Trainer is set to be debug mode, which may affect the performance")
            set_verb_level("debug")

        # setup for distributed
        self._use_sync_bn = use_sync_bn
        if is_distributed():
            if self._use_sync_bn:
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
                (self.logger.info if is_master() else self.logger.debug)(
                    "BNs of model are converted to nn.SyncBatchNorm")

            rank = get_local_rank()
            torch.cuda.set_device(rank)
            if get_global_rank() > 0:
                # to avoid overwriting
                quiet = True

        self.loss_f = loss_f
        self._verbose = not quiet

        # setup model
        if isinstance(model, nn.Module):
            self.model = model
        elif isinstance(model, dict):
            self.model = nn.ModuleDict(model)
            self.logger.debug(f"model is nn.ModuleDict of {self.model.keys()}")
        else:
            raise TypeError(f"Unknown type for `model`. Expected nn.Module or dict[str, Module], but got {type(model)}")

        if "cuda" in str(self.device):
            if not disable_auto_ddp:
                self.model.to(self.device)
            torch.backends.cudnn.benchmark = not disable_cudnn_benchmark
            self._cuda_nonblocking = not disable_cuda_nonblocking
            self.logger.debug(f"cuda: True, cudnn.benchmark: {not disable_cudnn_benchmark}, "
                              f"cuda.nonblocking: {not disable_cuda_nonblocking}")
        else:
            self._cuda_nonblocking = False
            # usually, this is not expected
            self.logger.info(f"cuda: False (torch.cuda.is_available()={torch.cuda.is_available()})")

        if is_distributed() and not disable_auto_ddp:
            dist_kwargs = dist_kwargs or {}
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[rank], **dist_kwargs)
            self.logger.debug(f"model converted to DistributedDataParallel at rank={rank}")

        # self.accessible_model is useful for e.g., checkpointing
        if isinstance(self.model, nn.parallel.DistributedDataParallel) or isinstance(self.model, nn.DataParallel):
            self.accessible_model = self.model.module
        else:
            self.accessible_model = self.model
        if disable_auto_ddp:
            self.logger.info("self.accessible_model need to be set manually")

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.reporter = None

        # called via property
        # _step and _epoch are set to -1 because they are incremented before each iteration and epoch
        self._step = -1
        self._epoch = -1
        self._is_train = True

        # to nest, leave=False (https://github.com/tqdm/tqdm/blob/master/examples/simple_examples.py#L19)
        self._tqdm = lambda x: x
        if self.verbose:
            self._tqdm = Partial(tqdm, ncols=tqdm_ncols, leave=False)
            set_tqdm_stdout_stderr()
            self.logger.debug("verbose: setup tqdm")
        else:
            self.logger.debug("quiet: no tqdm")

        # this needs to be here to access members from set_*,
        for k, v in kwargs.items():
            if hasattr(self, k):
                raise AttributeError(f"{self} already has {k}")
            if isinstance(v, torch.Tensor):
                v = v.to(self.device)
            if isinstance(v, nn.Module):
                v.to(self.device)
            setattr(self, k, v)
            self.logger.debug(f"trainer sets {k} as a new attribute")

        # setup optimizer and scheduler
        self.set_optimizer()
        self.set_scheduler()

        if reporters is not None and not isinstance(reporters, Iterable):
            reporters = [reporters]
        reporters = reporters or []

        if not any([isinstance(rep, TQDMReporter) for rep in reporters]):
            # if reporters not contain TQDMReporter
            reporters.append(TQDMReporter(ncols=tqdm_ncols))
        self.logger.debug(f"reporter is ready: {reporters}")
        self.reporter = ReporterList(reporters)

        if profile:
            default_prof_kwargs = dict(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA],
                                       schedule=torch.profiler.schedule(wait=1, warmup=1, active=10),
                                       on_trace_ready=torch.profiler.tensorboard_trace_handler("./profile"),
                                       with_stack=True)
            prof_kwargs = prof_kwargs or default_prof_kwargs
            self._profile_cm = torch.profiler.profile(**prof_kwargs)
            self.logger.warning("profiler is activated")
        else:
            class Dummy:
                def step(self): ...

            self._profile_cm = contextlib.nullcontext(Dummy())

    def tqdm(self,
             iter: Iterable
             ) -> Iterable:
        return self._tqdm(iter)

    @property
    def verbose(self
                ) -> bool:
        return self._verbose

    @property
    def step(self
             ) -> int:
        return self._step

    @property
    def epoch(self
              ) -> int:
        return self._epoch

    @property
    def is_train(self
                 ) -> bool:
        return self._is_train

    @property
    def history(self
                ) -> dict[str, list[float]]:
        return self.reporter.history

    @abstractmethod
    def iteration(self,
                  data: DataType
                  ) -> None:
        # Iteration
        pass

    def override_iteration(self,
                           new_iteration: Callable[[DataType], None]
                           ) -> None:
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

    def epoch_range(self,
                    epoch: int
                    ) -> TQDMReporter:
        tqdm_reporter = [rep for rep in self.reporter.reporters if isinstance(rep, TQDMReporter)][0]
        tqdm_reporter.set_iterator(range(epoch))
        return tqdm_reporter

    def infer_batch_size(self,
                         data: DataType
                         ) -> int:
        if isinstance(data, Tensor):
            batch_size = data.size(0)
        else:
            batch_size = data[0].size(0)
        return batch_size

    def _iteration(self,
                   data: DataType,
                   mode: str
                   ) -> None:
        """ Iteration level training loop

        :param data: should be TensorTuple
        :param mode: train, test or val
        :return:
        """

        data = self.data_preprocess(data)
        batch_size = self.infer_batch_size(data)
        self.reporter.set_batch_size(batch_size)
        if self._is_debug and batch_size == 1 and self.is_train:
            if any([isinstance(m, nn.modules.batchnorm._BatchNorm) for m in self.accessible_model.modules()]):
                warnings.warn("BatchNorm exists, while batch size is 1", RuntimeWarning)

        with torch.autograd.set_detect_anomaly(self._is_debug) and self._profile_cm as p:
            self.iteration(data)
            p.step()

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
              mode: str
              ) -> None:

        for data in self.tqdm(data_loader):
            if self.is_train:
                # increment step here for `callbacks`
                self._step += 1
            self._iteration(data, mode)

        self.reporter.report(self.epoch, mode)
        self.logger.debug(f"epoch {self.epoch} finished")

    def data_preprocess(self,
                        data: DataType
                        ) -> DataType:

        return TensorTuple(data).to(self.device, non_blocking=self._cuda_nonblocking)

    def train(self,
              data_loader: Iterable or DataLoader,
              mode: str = "train"
              ) -> None:
        """ Training the model for an epoch.

        :param data_loader:
        :param mode: Name of this loop. Default is `train`. Passed to callbacks.
        """

        self._is_train = True
        self._epoch += 1
        # For distributed training
        if isinstance(data_loader, DataLoader) and hasattr(data_loader.sampler, "set_epoch"):
            data_loader.sampler.set_epoch(self.epoch)
            self.logger.debug("set_epoch to the sampler")
        self.model.train()

        if hasattr(self.loss_f, "train"):
            self.loss_f.train()

        self._loop(data_loader, mode=mode)

        if self._is_debug:
            for name, param in self.accessible_model.named_parameters():
                self.reporter.add_histogram(name, param, self.epoch)

    def test(self,
             data_loader: Iterable or DataLoader,
             mode: str = "test"
             ) -> None:
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
            val_loaders: Iterable or DataLoader or dict[str, Iterable or DataLoader],
            total_iterations: int,
            val_intervals: int
            ) -> None:

        """ Train the model for a given iterations. This module is almost equal to ::

            for ep in range(total_iterations):
                trainer.train(train_loader)
                for k, v in val_loaders.items():
                    trainer.test(v, k)

        :param train_loader:
        :param val_loaders:
        :param total_iterations:
        :param val_intervals:
        :return:
        """

        class ProxyLoader(object):
            def __init__(self, loader):
                self.loader = loader
                self._epoch = 0

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
                    self._epoch += 1
                    if hasattr(self.loader, 'sampler') and hasattr(self.loader.sampler, 'set_epoch'):
                        self.loader.sampler.set_epoch(self._epoch)

        train_loader = ProxyLoader(train_loader)
        if not isinstance(val_loaders, dict) and (isinstance(val_loaders, Iterable) or
                                                  isinstance(val_loaders, DataLoader)):
            val_loaders = {'val': val_loaders}

        for ep in self.epoch_range(total_iterations // val_intervals):
            self.train(train_loader)
            for name, loader in val_loaders.items():
                self.test(loader, name)

    def exit(self):
        self.reporter.exit()

    def set_optimizer(self
                      ) -> None:
        """ Set optimizer(s) for model(s). You can override as::

            class YourTrainer(TrainerBase):
                def set_optimizer(self):
                    self.optimizer = torch.optim.SGD(self.model.parameters())

        :return:
        """

        optimizer = self.optimizer
        if isinstance(optimizer, Optimizer) or optimizer is None:
            self.optimizer = optimizer

        elif isinstance(optimizer, Partial):
            if not issubclass(optimizer.func, Optimizer):
                raise TypeError(f"`optimizer.func` is expected to be subclass of `Optimizer`"
                                f" but got {type(optimizer.func)}")
            self.optimizer = optimizer(self.model.parameters())

        elif isinstance(optimizer, dict):
            if not isinstance(self.model, nn.ModuleDict):
                raise TypeError("When `optimizer` is `dict`, `model` also needs to be `dict` or `nn.ModuleDict`")

            if isinstance(list(optimizer.values())[0], Partial):
                optimizer = {k: v(self.model[k].parameters()) for k, v in optimizer.items() if v is not None}
            self.optimizer = StepDict(Optimizer, **optimizer)

        else:
            raise TypeError(f"Unexpected type {type(optimizer)} for `optimizer`")

    def set_scheduler(self
                      ) -> None:
        """ Set scheduler(s) for optimizer(s). You can override as ::

            class YourTrainer(TrainerBase):
                def set_scheduler(self):
                    self.scheduler = torch.optim.lr_scheduler.Foo(self.optimizer)

        :return:
        """

        scheduler = self.scheduler
        if scheduler is not None and self.optimizer is None:
            raise TypeError("Optimizer is not set, so scheduler cannot be set")

        if isinstance(scheduler, Scheduler) or scheduler is None:
            self.scheduler = scheduler

        elif isinstance(scheduler, Partial):
            self.scheduler = scheduler(self.optimizer)

        elif isinstance(scheduler, dict):
            if not isinstance(self.optimizer, StepDict):
                raise TypeError("When `scheduler` is `dict`, `optimizer` is also needs to be `dict`")

            _scheduler = {}
            for k, v in scheduler.items():
                if isinstance(v, Partial):
                    v = v(self.optimizer[k])
                _scheduler[k] = v
            self.scheduler = StepDict(Scheduler, **_scheduler)

        else:
            raise TypeError(f"Unexpected type {type(scheduler)} for `scheduler`")

        if self.verbose and hasattr(self.scheduler, 'verbose'):
            self.scheduler.verbose = True


class SupervisedTrainer(TrainerBase):
    """ A simple trainer for supervised image classification. It only accepts single model. AMP-ready.

    """

    def __init__(self,
                 model: nn.Module,
                 optimizer: Optimizer,
                 loss_f: Callable,
                 *,
                 reporters: _ReporterBase or list[_ReporterBase] = None,
                 scheduler: Scheduler = None,
                 quiet=False,
                 disable_cudnn_benchmark=False,
                 data_parallel=False,
                 use_amp=False,
                 use_channel_last=False,
                 report_accuracy_topk: int or list[int] = None,
                 update_scheduler_iter: bool = False,
                 use_larc: bool = False,
                 grad_accum_steps: int = None,
                 **kwargs):
        if isinstance(model, dict):
            raise TypeError(f"{type(self)} does not support dict model")
        self.use_larc = use_larc
        super(SupervisedTrainer, self).__init__(model, optimizer, loss_f, reporters=reporters, scheduler=scheduler,
                                                quiet=quiet, disable_cudnn_benchmark=disable_cudnn_benchmark,
                                                **kwargs)

        if data_parallel and not isinstance(self.model, nn.DataParallel) and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            self.model.to(self.device)
            self.logger.info("model converted to DataParallel")

        self._use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=self._use_amp)
        if self._use_amp:
            (self.logger.info if is_master() else self.logger.debug)("AMP is activated")
        self._use_channel_last = use_channel_last
        if self._use_channel_last:
            self.logger.warning("channel_last format is an experimental feature")
            self.model = self.model.to(memory_format=torch.channels_last)
        if report_accuracy_topk is not None:
            if not isinstance(report_accuracy_topk, Iterable):
                report_accuracy_topk = [report_accuracy_topk]
            report_accuracy_topk = [k for k in report_accuracy_topk if k != 1]
        self._report_topk = report_accuracy_topk
        self.update_scheduler_iter = update_scheduler_iter & (scheduler is not None)
        if self.update_scheduler_iter:
            self.logger.info("scheduler is set to be updated after every iteration")
        else:
            self.logger.debug("self.update_scheduler_iter=False. Update scheduler manually")

        if grad_accum_steps is not None and grad_accum_steps <= 1:
            raise ValueError('grad_accum_steps should be equal or larger than 2.')
        self.grad_accum_steps = 1 if grad_accum_steps is None else grad_accum_steps
        if self.grad_accum_steps > 1:
            self.logger.info('Gradient accumulation is activated')

    def iteration(self,
                  data: tuple[Tensor, Tensor]
                  ) -> None:
        input, target = data

        if self.is_train:
            loss = 0

            for input, target in zip(input.chunk(self.grad_accum_steps), target.chunk(self.grad_accum_steps)):
                with torch.cuda.amp.autocast(self._use_amp):
                    output = self.model(input)
                    _loss = self.loss_f(output, target) / self.grad_accum_steps
                    loss += _loss.detach()
                    # this code supports both AMP and non AMP
                    self.scaler.scale(_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            if self.update_scheduler_iter:
                self.scheduler.step()

        else:
            # test time
            with torch.cuda.amp.autocast(self._use_amp):
                output = self.model(input)
                loss = self.loss_f(output, target)
        if self._is_debug and torch.isnan(loss):
            self.logger.warning("loss is NaN")

        self.reporter.add('accuracy', accuracy(output, target))
        self.reporter.add('loss', loss.detach_())
        if self._report_topk is not None:
            for top_k in self._report_topk:
                self.reporter.add(f'accuracy@{top_k}', accuracy(output, target, top_k))

    def data_preprocess(self,
                        data: tuple[Tensor, Tensor]
                        ) -> tuple[Tensor, Tensor]:
        input, target = data
        return (input.to(self.device, non_blocking=self._cuda_nonblocking,
                         memory_format=torch.channels_last if self._use_channel_last
                         else torch.preserve_format),
                target.to(self.device, non_blocking=self._cuda_nonblocking))

    def state_dict(self
                   ) -> dict[str, Any]:

        return {'model': self.accessible_model.state_dict(),
                'optim': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'epoch': self.epoch,
                'use_sync_bn': self._use_sync_bn,
                'use_amp': self._use_amp}

    def load_state_dict(self,
                        state_dict: dict[str, Any]
                        ) -> None:
        self.accessible_model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optim'])
        self.scheduler.load_state_dict(state_dict['scheduler'])
        self.scheduler.last_epoch = state_dict['epoch']
        self._epoch = state_dict['epoch']
        self._use_sync_bn = state_dict['use_sync_bn']
        self._use_amp = state_dict['use_amp']

    def set_scheduler(self
                      ) -> None:
        super().set_scheduler()
        if self.use_larc:
            # scheduler expects optimizer is Optimizer, but LARC is not
            self.optimizer = LARC(self.optimizer)
