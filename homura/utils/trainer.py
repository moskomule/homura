import collections
import torch
from torch.autograd import Variable
from .reporter import TQDMReporter, ListReporter


class Trainer(object):

    def __init__(self, model, optimizer, loss_f, reporters, scheduler=None,
                 log_freq=10, verb=True, metrics="default",
                 use_cuda=True, use_cudnn_bnenchmark=True, report_parameters=False, **kwargs):
        self.model = model
        self._optimizer = optimizer
        self._loss_f = loss_f
        if not isinstance(reporters, collections.Iterable):
            reporters = [reporters]
        self._reporters = ListReporter(reporters)
        self._scheduler = scheduler
        self._steps = {"train": 0, "test": 0}
        self._epochs = 0
        self._log_freq = log_freq
        self._verb = verb
        self._metrics = {"accuracy": self.correct} if metrics == "default" else metrics
        self._use_cuda = use_cuda and torch.cuda.is_available()
        if self._use_cuda:
            if use_cudnn_bnenchmark:
                torch.backends.cudnn.benchmark = True
            self.model.cuda()

        self._report_parameters = report_parameters
        for k, v in kwargs.items():
            if hasattr(self, k):
                raise AttributeError(f"{self} already has {k}")
            setattr(self, k, v)

    def _loop(self, data_loader, is_train=True):
        mode = "train" if is_train else "test"
        data_loader = TQDMReporter(data_loader) if self._verb else data_loader

        loop_loss = 0
        loop_metrics = {k: 0 for k in self._metrics.keys()}

        for data in data_loader:
            loss, metrics = self._iteration(data, is_train)
            loop_loss += loss
            for k in loop_metrics.keys():
                loop_metrics[k] += metrics[k]
            self._steps[mode] += 1

            if is_train and self._steps[mode] % self._log_freq == 0:
                self._reporters.add_scalar("iter_train_loss", loss, self._steps[mode])

        for reporter in self._reporters:
            reporter.add_scalar(f"epoch_{mode}_loss", loop_loss / len(data_loader), self._epochs)

        for name, metrics in loop_metrics.items():
            if metrics:
                self._reporters.add_scalar(f"{mode}_{name}", metrics / len(data_loader), self._epochs)

    def _iteration(self, data, is_train):
        input, target = data
        input, target = self.variable(input, volatile=not is_train), self.variable(target, volatile=not is_train)
        output = self.model(input)
        loss = self._loss_f(output, target)
        if is_train:
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
        loss = loss.data[0]
        metrics = {k: fn(output, target) for k, fn in self._metrics.items()}
        return loss, metrics

    def train(self, data_loader):
        self.model.train()
        self._loop(data_loader)
        if self._scheduler is not None:
            self._scheduler.step()
        if self._report_parameters:
            for name, param in self.model.named_parameters():
                    self._reporters.add_histogram(name, param, self._epochs, bins="sqrt")
        self._epochs += 1

    def test(self, data_loader):
        self.model.eval()
        self._loop(data_loader, is_train=False)

    def run(self, epochs, train_data, test_data):
        try:
            for ep in range(1, epochs + 1):
                if self._verb:
                    print(f"epochs: {ep}")
                self.train(train_data)
                self.test(test_data)
        except KeyboardInterrupt:
            print("\ninterrupted")
        finally:
            self._reporters.close()

    @staticmethod
    def correct(input, target):
        return (input.max(dim=1)[1] == target).data.mean()[0]

    def variable(self, t, **kwargs):
        if self._use_cuda:
            t = t.cuda()
        return Variable(t, **kwargs)
