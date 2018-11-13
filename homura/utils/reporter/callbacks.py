from typing import Iterable
from .wrapper import TQDMWrapper, VisdomWrapper, TensorBoardWrapper
from ..callbacks import Callback, CallbackList
from .._vocabulary import *


class _Reporter(Callback):
    def __init__(self, wrapper, callbacks: Iterable[Callback], report_freq: int, wrapper_args: dict):
        self.base_wrapper = wrapper(**wrapper_args)
        self.callback = CallbackList(*callbacks)
        self.report_freq = report_freq
        self.report_params_freq = -1
        self.report_params = False

    def add_callbacks(self, *callbacks):
        self.callback._callbacks += list(callbacks)

    def after_iteration(self, data: dict):
        results = self.callback.after_iteration(data)

        if (data[STEP] % self.report_freq == 0) and self.report_freq > 0:
            for k, v in results.items():
                self.base_wrapper.add_scalar(v, name=f"{STEP}_{k}", idx=data[STEP])

            if self.report_params and (data[STEP] % self.report_params_freq == 0):
                self._add_params(data[MODEL], data[STEP])

    def after_epoch(self, data: dict):
        results = self.callback.after_epoch(data)
        for k, v in results.items():
            self.base_wrapper.add_scalar(v, name=k, idx=data[EPOCH])

        if self.report_params and (self.report_params_freq == -1):
            self._add_params(data[MODEL], data[EPOCH])

    def close(self):
        self.base_wrapper.close()
        self.callback.close()

    def report_parameters(self, report_freq=-1):
        self.report_params = True
        self.report_params_freq = report_freq

    def _add_params(self, params, idx):
        for name, param in params.named_parameters():
            self.base_wrapper.add_histogram(param, name, idx)


class TQDMReporter(_Reporter):
    def __init__(self, iterator, callbacks, save_dir=None, report_freq=-1):
        super(TQDMReporter, self).__init__(TQDMWrapper, callbacks, report_freq,
                                           {"iterator": iterator, "save_dir": save_dir})

    def __iter__(self):
        for x in self.base_wrapper:
            yield x

    def __len__(self):
        return len(self.base_wrapper)


class VisdomReporter(_Reporter):
    def __init__(self, callbacks, port=6006, save_dir=None, report_freq=-1):
        super(VisdomReporter, self).__init__(VisdomWrapper, callbacks, report_freq,
                                             {"port": port, "save_dir": save_dir})


class TensorboardReporter(_Reporter):
    def __init__(self, callbacks, save_dir=None, report_freq=-1):
        super(TensorboardReporter, self).__init__(TensorBoardWrapper, callbacks, report_freq,
                                                  {"save_dir": save_dir})
