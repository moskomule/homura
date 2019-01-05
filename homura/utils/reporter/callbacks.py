from abc import ABCMeta
from numbers import Number
from typing import Iterable, Mapping
import warnings
import torch

from .wrapper import TQDMWrapper, VisdomWrapper, TensorBoardWrapper
from .._vocabulary import *
from ..callbacks import Callback, CallbackList


class Reporter(Callback, metaclass=ABCMeta):
    __param_reportable = False
    __image_reportable = False

    def __init__(self, wrapper, callbacks: Iterable[Callback], report_freq: int, wrapper_args: Mapping):
        self.base_wrapper = wrapper(**wrapper_args)
        self.callback = CallbackList(*callbacks)
        self._report_freq = report_freq
        self._report_params = False
        self._report_params_freq = -1
        self._report_images = False
        self._report_images_keys = []
        self._report_images_freq = -1
        self._iteration = 0
        self._tt_iteration_report_done = False

    def add_memo(self, text: str, *, name="memo", index=0):
        if name == "memo":
            # to avoid collision
            name += str(hash(text))[:5]
        self.base_wrapper.add_text(text, name, index)

    def _report_value(self, v, name, idx):
        if isinstance(v, Number) or isinstance(v, torch.Tensor):
            self.base_wrapper.add_scalar(v, name=name, idx=idx)
        elif isinstance(v, str):
            self.base_wrapper.add_text(v, name=name, idx=idx)
        else:
            raise NotImplementedError(f"Cannot report type :{type(v)}")

    def before_iteration(self, data: Mapping):
        self.callback.before_iteration(data)
        self._iteration += 1
        self._tt_iteration_report_done = False

    def after_iteration(self, data: Mapping):
        results = self.callback.after_iteration(data)
        mode = data[MODE]
        step = data[STEP]

        if (step % self._report_freq == 0) and self._report_freq > 0:
            for k, v in results.items():
                name = f"{STEP}_{k}"
                self._report_value(v, name, step)

        # to avoid repeatedly reporting when not training mode
        if mode != TRAIN and self._tt_iteration_report_done:
            warnings.warn("Parameters and images are reported once in a testing loop!", RuntimeWarning)
            return None

        if self._report_params and (step % self._report_params_freq == 0) and not (self._report_params_freq == -1):
            self.report_params(data[MODEL], step)

        if self._report_images and (step % self._report_images_freq == 0) and not (self._report_images_freq == -1):
            for key in self._report_images_keys:
                if data.get(key) is not None:
                    self.report_images(data[key], f"{key}_{mode}", step)

        if mode != TRAIN:
            self._tt_iteration_report_done = True

    def before_epoch(self, data: Mapping):
        self.callback.before_epoch(data)

    def after_epoch(self, data: Mapping):
        results = self.callback.after_epoch(data)
        mode = data[MODE]
        epoch = data[EPOCH]

        for k, v in results.items():
            self._report_value(v, k, epoch)

        if self._report_params and (self._report_params_freq == -1):
            self.report_params(data[MODEL], epoch)

        if self._report_images and (self._report_images_freq == -1):
            for key in self._report_images_keys:
                if data.get(key) is not None:
                    self.report_images(data[key], f"{key}_{mode}", epoch)

    def close(self):
        self.base_wrapper.close()
        self.callback.close()

    def enable_report_params(self, report_freq=-1):
        self._report_params = True
        self._report_params_freq = report_freq

    def disable_report_params(self):
        self._report_params = False

    def report_params(self, model, idx=None):
        idx = self._iteration if idx is None else idx
        for name, param in model.named_parameters():
            self.base_wrapper.add_histogram(param, name, idx)

    def enable_report_images(self, keys, report_freq=-1):
        self._report_images_keys += list(keys)
        self._report_images = True
        self._report_images_freq = report_freq

    def disable_report_images(self, keys=None):
        if keys is None:
            self._report_images = False
        else:
            raise NotImplementedError

    def report_images(self, image_tensor, name, idx=None):
        idx = self._iteration if idx is None else idx
        self.base_wrapper.add_images(image_tensor, name, idx)


class TQDMReporter(Reporter):
    def __init__(self, iterator, callbacks, save_dir=None, report_freq=-1):
        super(TQDMReporter, self).__init__(TQDMWrapper, callbacks, report_freq,
                                           {"iterator": iterator, "save_dir": save_dir})

    def __iter__(self):
        for x in self.base_wrapper:
            yield x

    def __len__(self):
        return len(self.base_wrapper)


class VisdomReporter(Reporter):
    __param_reportable = False
    __image_reportable = True

    def __init__(self, callbacks, port=6006, save_dir=None, report_freq=-1):
        super(VisdomReporter, self).__init__(VisdomWrapper, callbacks, report_freq,
                                             {"port": port, "save_dir": save_dir})


class TensorboardReporter(Reporter):
    __param_reportable = True
    __image_reportable = True

    def __init__(self, callbacks, save_dir=None, report_freq=-1):
        super(TensorboardReporter, self).__init__(TensorBoardWrapper, callbacks, report_freq,
                                                  {"save_dir": save_dir})
