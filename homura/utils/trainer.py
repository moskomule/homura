from typing import Callable

import torch
from torch import nn
from torch.optim import Optimizer
from torch.autograd import Variable
from .reporter import TQDMReporter
from .callbacks import CallbackList, Callback
from ._vocabulary import *


class Trainer(object):

    def __init__(self, model: nn.Module, optimizer: Optimizer, loss_f: Callable, *,
                 callbacks: Callback = None, scheduler=None, verb=True,
                 use_cuda=True, use_cudnn_bnenchmark=True, **kwargs):
        self._model = model
        self._optimizer = optimizer
        self._loss_f = loss_f
        self._callbacks = CallbackList(callbacks)
        self._scheduler = scheduler
        self._step = 0
        self._epoch = 0
        self._verb = verb
        self._use_cuda = use_cuda and torch.cuda.is_available()
        if self._use_cuda:
            if use_cudnn_bnenchmark:
                torch.backends.cudnn.benchmark = True
            self._model.cuda()

        for k, v in kwargs.items():
            if hasattr(self, k):
                raise AttributeError(f"{self} already has {k}")
            setattr(self, k, v)

    def _iteration(self, data, is_train, name):
        self._callbacks.start_epoch({MODEL: self._model,
                                     STEP: self._step,
                                     NAME: name,
                                     TRAINER: self})
        input, target = data
        input = self.to_device(input, volatile=not is_train)
        target = self.to_device(target, volatile=not is_train)
        output = self._model(input)
        loss = self._loss_f(output, target)
        if is_train:
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
        self._callbacks.end_iteration({OUTPUT: output,
                                       TARGET: target,
                                       MODEL: self._model,
                                       LOSS: loss.data[0],
                                       STEP: self._step,
                                       NAME: name,
                                       TRAINER: self})

    def _loop(self, data_loader, is_train, name):
        self._callbacks.start_epoch({MODEL: self._model,
                                     NAME: name,
                                     TRAINER: self})
        data_loader = TQDMReporter(data_loader) if self._verb else data_loader

        for data in data_loader:
            self._iteration(data, is_train, name)
            if is_train:
                self._step += 1
        self._callbacks.end_epoch({MODEL: self._model,
                                   EPOCH: self._epoch,
                                   NAME: name,
                                   ITER_PER_EPOCH: len(data_loader),
                                   TRAINER: self})

    def train(self, data_loader):
        self._model.train()
        self._loop(data_loader, is_train=True, name=TRAIN)
        if self._scheduler is not None:
            self._scheduler.step()
        self._epoch += 1

    def test(self, data_loader, name=TEST):
        self._model.eval()
        self._loop(data_loader, is_train=False, name=name)

    def run(self, epochs, train_data, test_data):
        try:
            for ep in range(1, epochs + 1):
                self.train(train_data)
                self.test(test_data)
            self._callbacks.end_all({MODEL: self._model,
                                     OPTIMIZER: self._optimizer,
                                     TRAINER: self})

        except KeyboardInterrupt:
            print("\ninterrupted")
        finally:
            self._callbacks.close()

    def to_device(self, t, **kwargs):
        if self._use_cuda:
            t = t.cuda()
        return Variable(t, **kwargs)
