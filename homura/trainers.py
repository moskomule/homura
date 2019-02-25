from abc import ABCMeta, abstractmethod
from pathlib import Path
from types import MethodType
from typing import Callable, Iterable, Dict, Mapping, Tuple, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import homura
from homura.liblog import get_logger
from homura.lr_scheduler import LRScheduler
from homura.optim import Optimizer
from .callbacks import Callback
from .utils.containers import TensorTuple, Map, StepDict
from .utils.miscs import check_path
from .utils.runner import Runner
from .utils._vocabulary import *

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
            optimizer.set_model(self.model.parameters())
            self.optimizer = optimizer.optim
        elif isinstance(optimizer, torch.optim.Optimizer):
            self.optimizer = optimizer
        elif isinstance(optimizer, dict):
            if not isinstance(model, dict):
                raise TypeError(f"model is not dict but optimizer is dict!")
            self.optimizer = StepDict(torch.optim.Optimizer)
            # self.model is nn.ModuleDict
            for k, opt in optimizer.items():
                m = self.model._modules.get(k)
                if m is None:
                    raise KeyError(f"No such key {k} in model!")
                if opt is None:
                    self.optimizer[k] = None
                elif isinstance(opt, Optimizer):
                    self.optimizer[k] = opt.set_model(m.parameters())
        else:
            raise TypeError(f"{type(optimizer)}")
        self.logger.debug(f"Use optimizer: {self.optimizer.__class__.__name__}")

        # set scheduler(s)
        if scheduler is None:
            self.scheduler = None
        elif isinstance(scheduler, LRScheduler):
            scheduler.set_optimizer(self.optimizer)
            self.scheduler = scheduler.scheduler
        elif isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler):
            self.scheduler = scheduler
        elif isinstance(scheduler, dict):
            if not isinstance(optimizer, dict):
                raise TypeError(f"optimizer is not dict but scheduler is dict!")
            self.scheduler = StepDict(torch.optim.lr_scheduler._LRScheduler)
            for k, schdlr in scheduler.items():
                opt = self.optimizer.get(k)
                if schdlr is None:
                    self.scheduler[k] = None
                self.scheduler[k] = schdlr.set_optimizer(opt)
        else:
            raise TypeError(f"{type(scheduler)}")
        self.logger.debug(f"Use scheduler: {self.scheduler.__class__.__name__}")
        self._update_scheduler_by_epoch = update_scheduler_by_epoch

        self.loss_f = loss_f
        self._verb = verb

        # called via property
        self._step = 0
        self._epoch = 0
        self._is_train = True

        _map_base = {MODEL: self.model,
                     OPTIMIZER: self.optimizer,
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
        """
        iteration part, user can override via duck typing or override_iteration

        :param data: data used during a iteration
        :return: loss, output
        """

    def override_iteration(self, new_iteration: Callable):
        """ Override iteration method ::

            def new_iteration(trainer, inputs):
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

    def _iteration(self, data: Tuple[torch.Tensor], mode: str):
        """ iteration level training loop for backend

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
            loss, output = TensorTuple(results).to(CPU)
            results = dict(loss=loss, output=output)
            self._iteration_map.update(**results)
        else:
            self._iteration_map.update(**results.to(CPU))
        self._iteration_map[DATA] = data.to(CPU)
        with torch.no_grad():
            self._callbacks.after_iteration(self._iteration_map)
        # clean up
        self._iteration_map.pop(DATA)
        for k in results.keys():
            self._iteration_map.pop(k)

    def _loop(self, data_loader: DataLoader, mode: str):
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
            self._iteration(data, mode)
            if self.is_train:
                self._step += 1

        with torch.no_grad():
            self._callbacks.after_epoch(self._epoch_map)
        self.logger.debug(f"epoch {self.epoch} finished")

    def train(self, data_loader: DataLoader):
        self._is_train = True
        self.model.train()
        with torch.enable_grad():
            self._loop(data_loader, mode=TRAIN)

        if self.scheduler is not None and self._update_scheduler_by_epoch:
            self.scheduler.step()
        self._epoch += 1

        # todo: try-except-finally like self.run
        # problem: tqdm may use an Exception for something?, which occurs an error.

    def test(self, data_loader: DataLoader, mode: str = TEST):
        self._is_train = False
        self.model.eval()
        with torch.no_grad():
            self._loop(data_loader, mode=mode)

    def run(self, epochs: int, train_data: DataLoader, test_data: DataLoader):
        try:
            for ep in range(1, epochs + 1):
                self.train(train_data)
                self.test(test_data)
            self._exit()

        except KeyboardInterrupt:
            print("\ninterrupted")
        finally:
            self._callbacks.close()
            exit()

    def _exit(self):
        with torch.no_grad():
            self._callbacks.after_all(self._all_map)

    def resume(self, path: str or Path):
        path = check_path(path)
        with path.open('rb') as f:
            loaded = torch.load(f)

        self.model.load_state_dict(loaded[MODEL])
        self.optimizer.load_state_dict(loaded[OPTIMIZER])
        self._step = loaded.get(STEP, 0)
        self._epoch = loaded.get(EPOCH, 0)
        self.logger.info(f"Resume training from {self.epoch}th epoch")


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
        return Map(loss=loss, output=output)


class FP16Trainer(TrainerBase):
    def __init__(self, model: nn.Module, optimizer: Optimizer, loss_f: Callable, *,
                 callbacks: Callback = None, scheduler: LRScheduler = None,
                 verb=True, use_cudnn_benchmark=True, static_loss_scale=None, **kwargs):
        if torch.cuda.is_available():
            raise RuntimeError("FP16Trainer requires CUDA backend!")
        if not homura.is_apex_available:
            raise RuntimeError("FP16Trainer requires apex!")
        if isinstance(model, dict):
            raise TypeError(f"{type(self)} does not support dict model")
        super(FP16Trainer, self).__init__(model, optimizer, loss_f, callbacks=callbacks, scheduler=scheduler,
                                          verb=verb, use_cudnn_benchmark=use_cudnn_benchmark, **kwargs)
        self.model.half()
        dynamic_loss_scale = (static_loss_scale is None)
        from apex.fp16_utils import FP16_Optimizer

        self.optimizer = FP16_Optimizer(self.optimizer, dynamic_loss_scale=dynamic_loss_scale,
                                        static_loss_scale=static_loss_scale)
        self.logger.info("Training with FP16")

    def iteration(self, data: Tuple[torch.Tensor]) -> Mapping[str, torch.Tensor]:
        input, target = data
        output = self.model(input)
        loss = self.loss_f(output, target)
        if self.is_train:
            self.optimizer.zero_grad()
            self.optimizer.backward(loss)
            self.optimizer.step()
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
    :param use_apex_ddp: use nvidia's `apex`.
    :param use_sync_bn:
    :param kwargs:
    """

    def __init__(self, model: nn.Module, optimizer: Optimizer, loss_f: Callable, *,
                 callbacks: Callback = None, scheduler: LRScheduler = None,
                 verb=True, use_cudnn_benchmark=True, backend="nccl", init_method="env://",
                 use_apex_ddp: bool = False, use_sync_bn: bool = False, **kwargs):
        import sys as python_sys
        from torch import distributed

        # should be used with torch.distributed.launch
        if "--local_rank" not in python_sys.argv:
            args = " ".join(python_sys.argv)
            raise RuntimeError(
                f"For distributed training, use python -m torch.distributed.launch "
                f"--nproc_per_node={torch.cuda.num_devices()} {args} ...")

        distributed.init_process_group(backend=backend, init_method=init_method)
        rank = distributed.get_rank()
        if rank != 0:
            # to avoid overwriting
            callbacks = None
            verb = False
        torch.cuda.set_device(rank)

        super(DistributedSupervisedTrainer, self).__init__(model, optimizer, loss_f, callbacks=callbacks,
                                                           scheduler=scheduler, verb=verb,
                                                           use_cudnn_benchmark=use_cudnn_benchmark,
                                                           use_cuda_nonblocking=True,
                                                           device=torch.device(GPU, rank), **kwargs)
        if use_apex_ddp:
            if not homura.is_apex_available:
                raise RuntimeError("arg. use_apex_ddp requires apex!")
            else:
                import apex.parallel

                self.model = apex.parallel.DistributedDataParallel(self.model)
                if use_sync_bn:
                    apex.parallel.convert_syncbn_model(self.model)
        else:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[rank])


# alias
Trainer = SupervisedTrainer
