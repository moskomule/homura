from pathlib import Path
from typing import Callable, Iterable

import torch
from torch import nn, optim

from ._vocabulary import *
from .callbacks import CallbackList, Callback
from .reporter.wrapper import TQDMWrapper

_optimizers = {"sgd": optim.SGD,
               "adam": optim.Adam,
               "adagrad": optim.Adagrad}


class Trainer(object):

    def __init__(self, model: nn.Module, optimizer: dict or optim.Optimizer, loss_f: Callable, *,
                 callbacks: Callback = None, scheduler=None, verb=True,
                 multi_gpu=False, use_cudnn_bnenchmark=True, **kwargs):
        """
        :param model: model to be trained
        :param optimizer: optimizer for the model. If dict,  {"name": "optimizer name", **kwargs}.
        :param loss_f: loss function
        :param callbacks: callbacks
        :param scheduler: learning rate scheduler
        :param verb:
        :param multi_gpu:
        :param use_cudnn_bnenchmark:
        :param kwargs:
        """

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        if self._device == "cuda":
            if use_cudnn_bnenchmark:
                torch.backends.cudnn.benchmark = True
            if multi_gpu:
                self.model = nn.DataParallel(model,
                                             device_ids=list(range(torch.cuda.device_count())))
            self.model.to(self._device)

        if isinstance(optimizer, dict):
            opt = optimizer.get("name", "").lower()
            if opt not in _optimizers.keys():
                raise NameError(f"Unknown optimizer: {opt} ({list(_optimizers.keys())})")
            self.optimizer = _optimizers[opt](self.model.parameters(),
                                              **{k: v for k, v in optimizer.items() if k != "name"})
        else:
            self.optimizer = optimizer

        self.loss_f = loss_f
        if isinstance(callbacks, CallbackList):
            self._callbacks = callbacks
        else:
            self._callbacks = CallbackList(callbacks)

        self._scheduler = scheduler
        self._step = 0
        self._epoch = 0
        self._verb = verb

        for k, v in kwargs.items():
            if hasattr(self, k):
                raise AttributeError(f"{self} already has {k}")
            setattr(self, k, v)

        self._start_iteration = {}
        self._end_iteration = {}
        self._start_epoch = {}
        self._end_epoch = {}
        self._end_all = {}

    def iteration(self, data: Iterable[torch.Tensor], is_train: bool) -> Iterable[torch.Tensor]:
        """
        iteration part, user can override
        :param data: data used during a iteration
        :param is_train:
        :return: loss, output
        """
        input, target = self.to_device(data)
        output = self.model(input)
        loss = self.loss_f(output, target)
        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss, output

    def register_start_iteration(self, name, data):
        self._start_iteration[name] = data

    def register_end_iteration(self, name, data):
        self._end_iteration[name] = data

    def register_start_epoch(self, name, data):
        self._start_epoch[name] = data

    def register_end_epoch(self, name, data):
        self._end_epoch[name] = data

    def register_end_all(self, name, data):
        self._end_all[name] = data

    def _iteration(self, data: Iterable[torch.Tensor], is_train: bool, name: str):
        with torch.no_grad():
            _start_iteration = {MODEL: self.model,
                                STEP: self._step,
                                NAME: name,
                                TRAINER: self}
            _start_iteration.update(self._start_iteration)
            self._callbacks.start_iteration(_start_iteration)
        loss, output = self.iteration(data, is_train)
        with torch.no_grad():
            _end_iteration = {OUTPUT: output.cpu(),
                              DATA: data,
                              MODEL: self.model,
                              LOSS: loss.data.item(),
                              STEP: self._step,
                              NAME: name,
                              TRAINER: self}
            _end_iteration.update(self._end_iteration)
            self._callbacks.end_iteration(_end_iteration)

    def _loop(self, data_loader, is_train: bool, name: str):
        with torch.no_grad():
            _start_epoch = {MODEL: self.model,
                            NAME: name,
                            TRAINER: self}
            _start_epoch.update(self._start_epoch)
            self._callbacks.start_epoch(_start_epoch)

        data_loader = TQDMWrapper(data_loader) if self._verb else data_loader

        for data in data_loader:
            self._iteration(data, is_train, name)
            if is_train:
                self._step += 1

        with torch.no_grad():
            _end_epoch = {MODEL: self.model,
                          OPTIMIZER: self.optimizer,
                          EPOCH: self._epoch,
                          NAME: name,
                          ITER_PER_EPOCH: len(data_loader),
                          TRAINER: self}
            _end_epoch.update(self._end_epoch)
            self._callbacks.end_epoch(_end_epoch)

    def train(self, data_loader):
        self.model.train()
        with torch.enable_grad():
            self._loop(data_loader, is_train=True, name=TRAIN)
        if self._scheduler is not None:
            self._scheduler.step()
        self._epoch += 1

    def test(self, data_loader, name=TEST):
        self.model.eval()
        with torch.no_grad():
            self._loop(data_loader, is_train=False, name=name)

    def run(self, epochs, train_data, test_data):
        try:
            for ep in range(1, epochs + 1):
                self.train(train_data)
                self.test(test_data)
            self._exit()

        except KeyboardInterrupt:
            print("\ninterrupted")
        finally:
            self._callbacks.close()

    def __enter__(self):
        return self

    def _exit(self):
        with torch.no_grad():
            _end_all = {MODEL: self.model,
                        OPTIMIZER: self.optimizer,
                        TRAINER: self}
            _end_all.update(self._end_all)
            self._callbacks.end_all(_end_all)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._exit()
        self._callbacks.close()

    def to_device(self, data, **kwargs):
        """
        Handle tuple of data
        :param data:
        :param kwargs:
        :return:
        """
        return (t.to(self._device, **kwargs) for t in data)

    def load(self, path, load_last=False):
        """
        Load a checkpoints saved by WeightSave callback
        :param path:
        :param load_last:
        """
        path = Path(path)
        if path.exists():
            if load_last and path.is_dir():
                last_checkpoint = max([p.name for p in path.glob("*.pkl")])
                path = path / last_checkpoint
            checkpoint = torch.load(path)
        else:
            raise FileNotFoundError(f"No file {str(path)}")

        self.model.load_state_dict(checkpoint[MODEL])
        self.optimizer.load_state_dict(checkpoint[OPTIMIZER])
        self._epoch = checkpoint[EPOCH]
