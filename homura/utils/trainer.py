from typing import Callable, Iterable
from pathlib import Path

import torch
from torch import nn, optim
from .reporter import TQDMReporter
from .callbacks import CallbackList, Callback
from ._vocabulary import *

_optimizers = {"sgd": optim.SGD,
               "adam": optim.Adam,
               "adagrad": optim.Adagrad}


class Trainer(object):

    def __init__(self, model: nn.Module, optimizer: dict or optim.Optimizer, loss_f: Callable, *,
                 callbacks: Callback = None, scheduler=None, verb=True,
                 use_cuda=True, use_cudnn_bnenchmark=True, **kwargs):
        """
        :param model: model to be trained
        :param optimizer: optimizer for the model. If dict,  {"name": "optimizer name", **kwargs}.
        :param loss_f: loss function
        :param callbacks: callbacks
        :param scheduler: learning rate scheduler
        :param verb:
        :param use_cuda:
        :param use_cudnn_bnenchmark:
        :param kwargs:
        """

        self.model = model
        self._use_cuda = use_cuda and torch.cuda.is_available()
        if self._use_cuda:
            self.model.cuda()
            if use_cudnn_bnenchmark:
                torch.backends.cudnn.benchmark = True

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

    def _iteration(self, data: Iterable[torch.Tensor], is_train: bool, name: str):
        with torch.no_grad():
            self._callbacks.start_epoch({MODEL: self.model,
                                         STEP: self._step,
                                         NAME: name,
                                         TRAINER: self})
        loss, output = self.iteration(data, is_train)
        with torch.no_grad():
            self._callbacks.end_iteration({OUTPUT: output,
                                           DATA: data,
                                           MODEL: self.model,
                                           LOSS: loss.data.item(),
                                           STEP: self._step,
                                           NAME: name,
                                           TRAINER: self})

    def _loop(self, data_loader, is_train: bool, name: str):
        with torch.no_grad():
            self._callbacks.start_epoch({MODEL: self.model,
                                         NAME: name,
                                         TRAINER: self})

        data_loader = TQDMReporter(data_loader) if self._verb else data_loader

        for data in data_loader:
            self._iteration(data, is_train, name)
            if is_train:
                self._step += 1

        with torch.no_grad():
            self._callbacks.end_epoch({MODEL: self.model,
                                       EPOCH: self._epoch,
                                       NAME: name,
                                       ITER_PER_EPOCH: len(data_loader),
                                       TRAINER: self})

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
            self._callbacks.end_all({MODEL: self.model,
                                     OPTIMIZER: self.optimizer,
                                     TRAINER: self})

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
        if self._use_cuda:
            return (t.cuda(**kwargs) for t in data)
        else:
            return data

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
