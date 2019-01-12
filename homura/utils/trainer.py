from abc import ABCMeta, abstractmethod
from types import MethodType
from typing import Callable, Iterable, Dict, Mapping, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from homura.lr_scheduler import LRScheduler
from homura.optim import Optimizer
from ._vocabulary import *
from .callbacks import CallbackList, Callback
from .containers import TensorTuple, Map, StepDict
from .reporter.wrapper import TQDMWrapper

__all__ = ["TrainerBase", "Trainer", "SupervisedTrainer", "DistributedSupervisedTrainer"]


class TrainerBase(metaclass=ABCMeta):

    def __init__(self, model: nn.Module or Dict[str, nn.Module],
                 optimizer: Optimizer or Dict[str, Optimizer],
                 loss_f: Callable or Dict[str, Callable], *,
                 callbacks: Callback or Iterable[Callable] = None,
                 scheduler: LRScheduler or Dict[LRScheduler] = None,
                 device: torch.device or str = None,
                 verb=True, use_cudnn_benchmark=True, use_cuda_nonblocking=False, **kwargs):
        """
        :param model: nn.Module or dict like {"generator": gen, "discriminator": dis}
        :param optimizer: homura.optimizer.Optimizer or dict like {"generator": Adam(lr=3e-4)}
        :param loss_f: loss function or dict of loss functions
        :param callbacks: callbacks or list of callbacks
        :param scheduler: homura.scheduler.LRScheduler or dict like {"generator": StepLR(10)}
        :param verb:
        :param use_cudnn_benchmark:
        :param use_cuda_nonblocking:
        :param kwargs:
        """

        if device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

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

        if "cuda" in str(self._device):
            if use_cudnn_benchmark:
                torch.backends.cudnn.benchmark = True
            self.model.to(self._device)
            self._cuda_nonblocking = use_cuda_nonblocking

        # set optimizer(s)
        if isinstance(optimizer, Optimizer):
            optimizer.set_model(self.model.parameters())
            self.optimizer = optimizer.optim
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
        elif optimizer is None:
            self.optimizer = None
        else:
            raise TypeError(f"{type(optimizer)}")

        # set scheduler(s)
        if scheduler is None:
            self.scheduler = None
        elif isinstance(scheduler, LRScheduler):
            scheduler.set_optimizer(self.optimizer)
            self.scheduler = scheduler.scheduler
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

        self.loss_f = loss_f

        # set callback(s)
        if isinstance(callbacks, CallbackList):
            self._callbacks = callbacks
        elif isinstance(callbacks, Iterable):
            self._callbacks = CallbackList(*callbacks)
        elif callbacks is None:
            # if callback is not set
            self._callbacks = Callback()
        else:
            raise TypeError(f"type(callbacks) should not be {type(callbacks)}!")

        self._step = 0
        self._epoch = 0
        self._verb = verb
        self._is_train = True

        # set kwargs
        for k, v in kwargs.items():
            if hasattr(self, k):
                raise AttributeError(f"{self} already has {k}")
            setattr(self, k, v)

        _map_base = {MODEL: self.model,
                     OPTIMIZER: self.optimizer,
                     TRAINER: self}
        self._iteration_map = Map(**_map_base.copy())
        self._epoch_map = Map(**_map_base.copy())
        self._all_map = Map(**_map_base.copy())

        self._callbacks.before_all(self._all_map)

    @abstractmethod
    def iteration(self, data: Iterable[torch.Tensor]) -> Mapping[str, torch.Tensor]:
        """
        iteration part, user can override
        :param data: data used during a iteration
        :return: loss, output
        """

    def override_iteration(self, new_iteration: Callable):
        """
        override iteration method
        >>> def new_iteration(trainer, inputs):
        >>>     ...
        >>>     results.loss = loss
        >>>     return results
        >>> trainer.update_iteration(new_iteration)

        :param new_iteration:
        :return:
        """
        setattr(self, "iteration", MethodType(new_iteration, self))

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
        """
        iteration level training loop backend
        :param data: should be TensorTuple
        :param mode: train, test or val
        :return:
        """
        self._iteration_map.update({STEP: self._step,
                                    MODE: mode})
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
        self._iteration_map[INPUTS] = data.to(CPU)
        with torch.no_grad():
            self._callbacks.after_iteration(self._iteration_map)
            # clean up
        self._iteration_map.pop(INPUTS)
        for k in results.keys():
            self._iteration_map.pop(k)

    def _loop(self, data_loader: DataLoader, mode: str):
        # handle epoch level training loop
        self._epoch_map.update({EPOCH: self._epoch,
                                MODE: mode,
                                ITER_PER_EPOCH: len(data_loader)})
        with torch.no_grad():
            self._callbacks.before_epoch(self._epoch_map)

        data_loader = TQDMWrapper(data_loader) if self._verb else data_loader

        for data in data_loader:
            data = TensorTuple(data).to(self._device, non_blocking=self._cuda_nonblocking)
            self._iteration(data, mode)
            if self.is_train:
                self._step += 1

        with torch.no_grad():
            self._callbacks.after_epoch(self._epoch_map)

    def train(self, data_loader: DataLoader):
        self._is_train = True
        self.model.train()
        with torch.enable_grad():
            self._loop(data_loader, mode=TRAIN)
        if isinstance(self.scheduler, dict):
            for scheduler in self.scheduler.values():
                if scheduler is not None:
                    scheduler.step()
        elif self.scheduler is not None:
            # lr_scheduler
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

    @property
    def is_train(self):
        return self._is_train


class SupervisedTrainer(TrainerBase):
    def __init__(self, model: nn.Module, optimizer: Optimizer, loss_f: Callable, *,
                 callbacks: Callback = None, scheduler: LRScheduler = None,
                 verb=True, use_cudnn_benchmark=True, data_parallel=False, **kwargs):
        if isinstance(model, dict):
            raise TypeError(f"{type(self)} does not support dict model")
        super(SupervisedTrainer, self).__init__(model, optimizer, loss_f, callbacks=callbacks, scheduler=scheduler,
                                                verb=verb, use_cudnn_benchmark=use_cudnn_benchmark, **kwargs)
        if data_parallel and not isinstance(self.model, nn.DataParallel):
            self.model = nn.DataParallel(self.model)
            self.model.to(self._device)

    def iteration(self, data: Tuple[torch.Tensor]) -> Mapping[str, torch.Tensor]:
        input, target = data
        output = self.model(input)
        loss = self.loss_f(output, target)
        if self.is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return Map(loss=loss, output=output)


class DistributedSupervisedTrainer(SupervisedTrainer):
    def __init__(self, model: nn.Module, optimizer: Optimizer, loss_f: Callable, *,
                 callbacks: Callback = None, scheduler: LRScheduler = None,
                 verb=True, use_cudnn_benchmark=True, backend="nccl", init_method="env://", **kwargs):
        from torch import distributed

        rank = distributed.get_rank()
        if rank != 0:
            # to avoid overwriting
            callbacks = None
            verb = False
        torch.cuda.set_device(rank)
        distributed.init_process_group(backend=backend, init_method=init_method)

        super(DistributedSupervisedTrainer, self).__init__(model, optimizer, loss_f, callbacks=callbacks,
                                                           scheduler=scheduler, verb=verb,
                                                           use_cudnn_benchmark=use_cudnn_benchmark,
                                                           use_cuda_nonblocking=True,
                                                           device=torch.device("cuda", rank), **kwargs)
        if not isinstance(model, nn.parallel.DistributedDataParallel):
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[rank])


# alias
Trainer = SupervisedTrainer
