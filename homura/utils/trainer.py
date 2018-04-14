import torch
from torch.autograd import Variable
from .reporter import TQDMReporter
from .callbacks import CallbackList
from ._vocabulary import V


class Trainer(object):

    def __init__(self, model, optimizer, loss_f, *,
                 callbacks=None, scheduler=None, verb=True,
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
        self._callbacks.start_epoch({V.MODEL: self._model,
                                     V.STEP: self._step,
                                     V.NAME: name,
                                     V.TRAINER: self})
        input, target = data
        input = self.variable(input, volatile=not is_train)
        target = self.variable(target, volatile=not is_train)
        output = self._model(input)
        loss = self._loss_f(output, target)
        if is_train:
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
        self._callbacks.end_iteration({V.OUTPUT: output,
                                       V.TARGET: target,
                                       V.MODEL: self._model,
                                       V.LOSS: loss.data[0],
                                       V.STEP: self._step,
                                       V.NAME: name,
                                       V.TRAINER: self})

    def _loop(self, data_loader, is_train, name):
        self._callbacks.start_epoch({V.MODEL: self._model,
                                     V.NAME: name,
                                     V.TRAINER: self})
        data_loader = TQDMReporter(data_loader) if self._verb else data_loader

        for data in data_loader:
            self._iteration(data, is_train, name)
            if is_train:
                self._step += 1
        self._callbacks.end_epoch({V.MODEL: self._model,
                                   V.EPOCH: self._epoch,
                                   V.NAME: name,
                                   V.ITER_PER_EPOCH: len(data_loader),
                                   V.TRAINER: self})

    def train(self, data_loader):
        self._model.train()
        self._loop(data_loader, is_train=True, name=V.TRAIN)
        if self._scheduler is not None:
            self._scheduler.step()
        self._epoch += 1

    def test(self, data_loader, name=V.TEST):
        self._model.eval()
        self._loop(data_loader, is_train=False, name=name)

    def run(self, epochs, train_data, test_data):
        try:
            for ep in range(1, epochs + 1):
                self.train(train_data)
                self.test(test_data)
            self._callbacks.end_all({V.MODEL: self._model,
                                     V.OPTIMIZER: self._optimizer,
                                     V.TRAINER: self})

        except KeyboardInterrupt:
            print("\ninterrupted")
        finally:
            self._callbacks.close()

    def variable(self, t, **kwargs):
        if self._use_cuda:
            t = t.cuda()
        return Variable(t, **kwargs)
