from abc import ABCMeta, abstractmethod
from pathlib import Path
from types import MethodType
from typing import Callable, Iterable, Dict, Mapping, Tuple, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from homura.liblog import get_logger
from homura.lr_scheduler import LRScheduler
from homura.optim import Optimizer
from .callbacks import Callback
from .utils._vocabulary import *
from .utils.containers import TensorTuple, Map, StepDict
from .utils.environment import is_distributed, get_global_rank, get_local_rank
from .utils.miscs import check_path
from .utils.runner import Runner

__all__ = ["TrainerBase", "Trainer", "SupervisedTrainer", "DistributedSupervisedTrainer"]


class TrainerBase(Runner, metaclass=ABCMeta):
    """

    :param model: nn.Module or dict like {"generator": gen, "discriminator": dis}
    :param optimizer: homura.optimizer.Optimizer or dict like {"generator": Adam(lr=3e-4)}
    :param loss_f: loss function or dict of loss functions
    :param callbacks: callbacks or list of callbacks
    :param scheduler: homura.scheduler.LRScheduler or dict like {"generator": StepLR(10)}
    :param update_scheduler_by_epoch: If True, update scheduler every epoch. If False and scheduler is given, scheduler
        is need to be update by user.
    :param device:
    :param verb:
    :param use_cudnn_benchmark:
    :param use_cuda_nonblocking:
    :param logger:
    :param kwargs:
    """

    def __init__(self, model: nn.Module or Dict[str, nn.Module],
                 optimizer: Optional[Optimizer or Dict[str, Optimizer] or torch.optim.Optimizer],
                 loss_f: Optional[Callable or Dict[str, Callable]], *,
                 callbacks: Optional[Callback or Iterable[Callable]] = None,
                 scheduler: Optional[LRScheduler or Dict[LRScheduler]] = None,
                 update_scheduler_by_epoch: bool = True,
                 device: Optional[torch.device or str] = None,
                 verb=True, use_cudnn_benchmark=True, use_cuda_nonblocking=False, logger=None, **kwargs):

        if logger is None:
            logger = get_logger(__name__)
        super(TrainerBase, self).__init__(model, callbacks, device, use_cudnn_benchmark, use_cuda_nonblocking, logger,
                                          **kwargs)

        # set optimizer(s)
        if optimizer is None:
            self.optimizer = None
        elif isinstance(optimizer, Optimizer):
            self.optimizer = optimizer.set_model(self.model.parameters())
        elif isinstance(optimizer, torch.optim.Optimizer):
            self.optimizer = optimizer
        elif isinstance(optimizer, dict):
            if not isinstance(model, dict):
                raise TypeError(f"model is not dict but optimizer is dict!")
            self.optimizer = StepDict(torch.optim.Optimizer)
            # self.model is nn.ModuleDict, then self.optimizer is StepDict
            for k, opt in optimizer.items():
                m = self.model._modules.get(k)
                if m is None:
                    raise KeyError(f"No such key {k} in model!")
                if opt is None:
                    self.optimizer[k] = None
                elif isinstance(opt, Optimizer):
                    self.optimizer[k] = opt.set_model(m.parameters())
                else:
                    raise TypeError(f"Unknown type: {type(opt)}")
        else:
            raise TypeError(f"Unknown type: {type(optimizer)}")
        self.logger.debug(f"Use optimizer: {self.optimizer.__class__.__name__}")

        # set scheduler(s)
        self.update_scheduler_by_epoch = update_scheduler_by_epoch
        self.update_scheduler(scheduler, update_scheduler_by_epoch)

        self.logger.debug(f"Use scheduler: {self.scheduler.__class__.__name__}")

        self.loss_f = loss_f
        self._verb = verb

        # called via property
        # _step and _epoch are set to -1 because they are incremented before each iteration and epoch!
        self._step = -1
        self._epoch = -1
        self._is_train = True

        _map_base = {MODEL: self.model,
                     OPTIMIZER: self.optimizer,
                     SCHEDULER: self.scheduler,
                     TRAINER: self}
        self._iteration_map = Map(**_map_base.copy())
        self._epoch_map = Map(**_map_base.copy())
        self._all_map = Map(**_map_base.copy())

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
    def iteration(self, data: Iterable[torch.Tensor]) -> Mapping[str, torch.Tensor]:
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

    def override_iteration(self, new_iteration: Callable):
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

    def register_before_iteration(self, name, data):
        self._iteration_map[name] = data

    def register_after_iteration(self, name, data):
        self._iteration_map[name] = data

    def register_before_epoch(self, name, data):
        self._epoch_map[name] = data

    def register_after_epoch(self, name, data):
        self._epoch_map[name] = data

    def register_before_all(self, name, data):
        self._all_map[name] = data

    def register_after_all(self, name, data):
        self._all_map[name] = data

    def _iteration(self,
                   data: Tuple[torch.Tensor],
                   mode: str):
        """ Iteration level training loop for backend

        :param data: should be TensorTuple
        :param mode: train, test or val
        :return:
        """

        self._iteration_map.update({STEP: self.step, MODE: mode})
        with torch.no_grad():
            self._callbacks.before_iteration(self._iteration_map)
        results = self.iteration(data)
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

        >>> with Trainer(...) as trainer:
        >>>     trainer.train(...)

        """

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit()

    def _loop(self,
              data_loader: DataLoader,
              mode: str):
        # handle epoch level training loop
        self._epoch_map.update({EPOCH: self.epoch,
                                STEP: self.step,
                                MODE: mode,
                                ITER_PER_EPOCH: len(data_loader)})
        with torch.no_grad():
            self._callbacks.before_epoch(self._epoch_map)

        data_loader = tqdm(data_loader, ncols=80) if self._verb else data_loader

        for data in data_loader:
            data = TensorTuple(data).to(self.device, non_blocking=self._cuda_nonblocking)
            if self.is_train:
                self._step += 1
            self._iteration(data, mode)

        with torch.no_grad():
            self._callbacks.after_epoch(self._epoch_map)
        self.logger.debug(f"epoch {self.epoch} finished")

    def train(self,
              data_loader: DataLoader,
              mode: str = TRAIN):
        """ Training loop.

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
             data_loader: DataLoader,
             mode: str = TEST):
        """ Non-training loop.

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
            epochs: int,
            train_data: DataLoader,
            test_data: DataLoader):
        try:
            for ep in range(1, epochs + 1):
                self.train(train_data)
                self.test(test_data)
            self.exit()

        except KeyboardInterrupt:
            print("\ninterrupted")
        finally:
            self._callbacks.close()
            exit()

    def exit(self):
        with torch.no_grad():
            self._callbacks.after_all(self._all_map)

    def resume(self,
               path: str or Path):
        """ Resume training from saved states by `homura.callbacks.WeightSave`.

        :param path:
        :return:
        """

        path = check_path(path)
        with path.open('rb') as f:
            loaded = torch.load(f)

        self.model.load_state_dict(loaded[MODEL])
        if loaded.get(OPTIMIZER) is not None:
            self.optimizer.load_state_dict(loaded[OPTIMIZER])
        if loaded.get(SCHEDULER) is not None:
            self.scheduler.load_state_dict(loaded[SCHEDULER])
        self._step = loaded.get(STEP, 0)
        self._epoch = loaded.get(EPOCH, 0)
        self.logger.info(f"Resume training from {self.epoch}th epoch")

    def update_scheduler(self,
                         scheduler: LRScheduler,
                         update_scheduler_by_epoch: bool = True):
        if scheduler is None:
            self.scheduler = None
        elif isinstance(scheduler, LRScheduler) and isinstance(self.optimizer, torch.optim.Optimizer):
            self.scheduler = scheduler.set_optimizer(self.optimizer)
        elif isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler):
            self.scheduler = scheduler
        elif isinstance(scheduler, dict):
            if not isinstance(self.optimizer, StepDict):
                raise TypeError(f"optimizer is not dict but scheduler is dict!")
            self.scheduler = StepDict(torch.optim.lr_scheduler._LRScheduler)
            for k, schdlr in scheduler.items():
                opt = self.optimizer.get(k)
                if schdlr is None:
                    self.scheduler[k] = None
                elif isinstance(schdlr, LRScheduler):
                    self.scheduler[k] = schdlr.set_optimizer(opt)
                else:
                    raise TypeError(f"Unknown type: {type(schdlr)}")
        else:
            raise TypeError(f"Unknown type: {type(scheduler)}")
        self.logger.debug(f"Use scheduler: {self.scheduler.__class__.__name__}")
        self.update_scheduler_by_epoch = update_scheduler_by_epoch


class SupervisedTrainer(TrainerBase):
    def __init__(self, model: nn.Module, optimizer: Optimizer, loss_f: Callable, *,
                 callbacks: Optional[Callback or Iterable[Callable]] = None, scheduler: Optional[LRScheduler] = None,
                 verb=True, use_cudnn_benchmark=True, data_parallel=False, **kwargs):
        if isinstance(model, dict):
            raise TypeError(f"{type(self)} does not support dict model")
        super(SupervisedTrainer, self).__init__(model, optimizer, loss_f, callbacks=callbacks, scheduler=scheduler,
                                                verb=verb, use_cudnn_benchmark=use_cudnn_benchmark, **kwargs)
        if data_parallel and not isinstance(self.model, nn.DataParallel):
            self.model = nn.DataParallel(self.model)
            self.model.to(self.device)

    def iteration(self, data: Tuple[torch.Tensor]) -> Mapping[str, torch.Tensor]:
        input, target = data
        output = self.model(input)
        loss = self.loss_f(output, target)
        if self.is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None and not self.update_scheduler_by_epoch:
                self.scheduler.step()
        return Map(loss=loss, output=output)


class DistributedSupervisedTrainer(SupervisedTrainer):
    """ Trainer with distributed functions

    :param model:
    :param optimizer:
    :param loss_f:
    :param callbacks:
    :param scheduler:
    :param verb:
    :param use_cudnn_benchmark:
    :param backend: "nccl" or "gloo"
    :param init_method:
    :param use_sync_bn:
    :param kwargs:
    """

    def __init__(self, model: nn.Module, optimizer: Optimizer, loss_f: Callable, *,
                 callbacks: Callback = None, scheduler: LRScheduler = None,
                 verb=True, use_cudnn_benchmark=True, backend="nccl", init_method="env://",
                 use_sync_bn: bool = False, enable_amp=False, **kwargs):
        if use_sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if enable_amp:
            from homura import is_apex_available

            if not is_apex_available:
                raise RuntimeError("apex not installed")

        import sys as python_sys
        from torch import distributed

        # should be used with torch.distributed.launch
        if not is_distributed:
            raise RuntimeError(
                f"For distributed training, use python -m torch.distributed.launch "
                f"--nproc_per_node={torch.cuda.device_count()} {' '.join(python_sys.argv)} ...")

        distributed.init_process_group(backend=backend, init_method=init_method)
        rank = get_local_rank()
        if get_global_rank() > 0:
            # to avoid overwriting
            verb = False
        torch.cuda.set_device(rank)

        super(DistributedSupervisedTrainer, self).__init__(model, optimizer, loss_f, callbacks=callbacks,
                                                           scheduler=scheduler, verb=verb,
                                                           use_cudnn_benchmark=use_cudnn_benchmark,
                                                           use_cuda_nonblocking=True,
                                                           device=torch.device(GPU, rank), **kwargs)

        self.loss_scaler = None
        if enable_amp:
            from apex import amp
            from apex.parallel import DistributedDataParallel

            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O2")
            self.model = DistributedDataParallel(self.model, delay_allreduce=True)
            self.loss_scaler = amp.scale_loss
        else:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[rank])

    def iteration(self, data: Tuple[torch.Tensor]) -> Mapping[str, torch.Tensor]:
        input, target = data
        output = self.model(input)
        loss = self.loss_f(output, target)
        if self.is_train:
            self.optimizer.zero_grad()
            if self.loss_scaler is not None:
                with self.loss_scaler(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            self.optimizer.step()

            if self.scheduler is not None and not self.update_scheduler_by_epoch:
                self.scheduler.step()
        return Map(loss=loss, output=output)


# alias
Trainer = SupervisedTrainer
