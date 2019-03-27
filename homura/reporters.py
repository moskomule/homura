from abc import ABCMeta
from logging import Logger
from typing import Iterable, Mapping, Optional, Union

import torch
from torch import nn

from .callbacks import Callback, CallbackList
from .utils._vocabulary import *
from .utils.environment import get_global_rank
from .utils.reporter_backends import TQDMWrapper, TensorBoardWrapper, LoggerWrapper, _num_elements, _WrapperBase


class Reporter(Callback, metaclass=ABCMeta):

    def __init__(self, base_wrapper: _WrapperBase,
                 callbacks: Union[Iterable[Callback], Callback],
                 report_freq: int = -1,
                 report_param_freq: int = 0,
                 report_image_freq: int = 0,
                 image_keys: Optional[Iterable[str]] = None):

        self.base_wrapper = base_wrapper
        self.callbacks = callbacks if isinstance(callbacks, Callback) else CallbackList(*callbacks)
        self.report_freq = report_freq
        self.report_param_freq = report_param_freq
        self.report_image_freq = report_image_freq
        self.image_keys = [] if image_keys is None else image_keys
        self._reportable = True
        # if distributed and not the master, do not report
        if get_global_rank() > 0:
            self._reportable = False

    def add_memo(self, text: str, *, name="memo", index=0):
        if name == "memo":
            # to avoid collision
            name += str(hash(text))[:5]
        if self._reportable:
            self.base_wrapper.add_text(text, name, index)

    def before_all(self, data: Mapping):
        self.callbacks.before_all(data)

    def after_all(self, data: Mapping):
        self.callbacks.after_all(data)

    def before_iteration(self, data: Mapping):
        self.callbacks.before_iteration(data)

    def after_iteration(self, data: Mapping):
        f"""
        :param data: Mapping. Requires to have at least {MODE}, {STEP}, {MODEL} as its key
        """
        results = self.callbacks.after_iteration(data)

        if self._reportable:
            mode = data[MODE]
            step = data[STEP]

            if self.report_freq > 0 and (step % self.report_freq == 0):
                self._report(results, mode, step)

            if self.report_image_freq > 0 and (step % self.report_image_freq == 0):
                for k in self.image_keys:
                    if data.get(k) is not None:
                        self._report_images(data[k], f"{mode}_{k}", step)

            if mode == TRAIN and self.report_param_freq > 0 and (step % self.report_param_freq == 0):
                self._report_params(data[MODEL], step)

    def before_epoch(self, data: Mapping):
        self.callbacks.before_epoch(data)

    def after_epoch(self, data: Mapping):
        f"""
        :param data: Mapping. Requires to have at least {MODE}, {EPOCH}, {MODEL} as its key
        """

        results = self.callbacks.after_epoch(data)

        if self._reportable:
            mode = data[MODE]
            epoch = data[EPOCH]

            self._report(results, mode, epoch)

            if self.report_image_freq < 0:
                for k in self.image_keys:
                    if data.get(k) is not None:
                        self._report_images(data[k], f"{mode}_{k}", epoch)

            if mode == TRAIN and self.report_param_freq < 0:
                self._report_params(data[MODEL], epoch)

    def _report(self, results: Mapping, mode: str, idx: int):
        for k, v in results.items():
            # images are processed by self._report_images
            if k in self.image_keys:
                continue
            if not isinstance(v, Mapping) and _num_elements(v) == 1:
                self.base_wrapper.add_scalar(v, k, idx)
            else:
                self.base_wrapper.add_scalars(v, k, idx)

    def _report_params(self, model: nn.Module, idx: int):
        for name, param in model.named_parameters():
            self.base_wrapper.add_histogram(param, name, idx)

    def _report_images(self, image: torch.Tensor, name: str, idx: int):
        self.base_wrapper.add_images(image, name, idx)

    def close(self):
        self.callbacks.close()
        self.base_wrapper.close()

    def enable_report_images(self, report_freq: int = -1, image_keys: Union[str, list] = None):
        if image_keys is not None:
            if isinstance(image_keys, str):
                image_keys = [image_keys]
            self.report_param_freq = report_freq
            self.image_keys += list(image_keys)
        else:
            raise ValueError("Argument image_keys should be specified!")

    def enable_report_params(self, report_freq: int = -1):
        self.report_param_freq = report_freq

    def disable_report_images(self, image_keys: Optional[Union[str, list]] = None):
        if image_keys is None:
            image_keys = self.image_keys
        elif isinstance(image_keys, str):
            image_keys = [image_keys]
        for k in image_keys:
            self.image_keys.pop(k)
        if len(self.image_keys) == 0:
            self.report_image_freq = 0

    def disable_report_params(self):
        pass


class TQDMReporter(Reporter):
    """ Reporter using tqdm. Use like ::

        with TQDMReporter(range(100), callbacks) as tq:
            trainer = ...
            for _ in tq:
                ...

    The output is like ::

        100%|█| 200/200 [1:01:10<00:00, 18.44s/it, loss_test=0.358, accuracy_test=0.919]

    :param iterator:
    :param callbacks:
    :param save_dir:
    :param report_freq:
    :param save_image_freq: If n>0, saves images every n iteration. if n==-1, every epoch.
        This may need large storage space.
    :param image_keys: keys for images.
    """

    def __init__(self, iterator: Iterable,
                 callbacks: Union[Iterable[Callback], Callback],
                 save_dir: Optional[str] = None,
                 report_freq: int = -1,
                 save_image_freq: int = 0,
                 image_keys: Optional[Iterable[str]] = None):
        super(TQDMReporter, self).__init__(TQDMWrapper(iterator=iterator, save_dir=save_dir), callbacks,
                                           report_freq=report_freq, report_image_freq=save_image_freq,
                                           image_keys=image_keys)

    def __iter__(self):
        for x in self.base_wrapper:
            yield x

    def __len__(self):
        return len(self.base_wrapper)

    def _report(self, results: Mapping, mode: str, idx: int):
        results = {k: float(v) for k, v in results.items() if not isinstance(v, Mapping) and _num_elements(v) == 1}
        self.base_wrapper.add_scalars(results, None, idx)


class LoggerReporter(Reporter):
    """ Something like this ::

        with LoggerReporter(...) as lr:
            trainer = ...
            for _ in range(100):
                ...

    Reports like this ::

        [homura.reporter|2019-02-30 27:31:20|INFO] [      1]                      loss_test 0.4413
        [homura.reporter|2019-02-30 27:31:20|INFO] [      1]                  accuracy_test 0.8631

    :param callbacks:
    :param save_dir:
    :param report_freq:
    :param save_image_freq: If n>0, saves images every n iteration. if n==-1, every epoch.
        This may need large storage space.
    :param image_keys: keys for images.
    """

    def __init__(self, callbacks: Union[Iterable[Callback], Callback],
                 save_dir: Optional[str] = None,
                 logger: Optional[Logger] = None,
                 report_freq: int = -1,
                 save_image_freq: int = 0,
                 image_keys: Optional[Iterable[str]] = None,
                 save_log: bool = False):
        super(LoggerReporter, self).__init__(LoggerWrapper(save_dir=save_dir, logger=logger, save_log=save_log),
                                             callbacks, report_freq,
                                             report_image_freq=save_image_freq, image_keys=image_keys)


class TensorboardReporter(Reporter):
    """ Reporter using Tensorboard. ::

        with TensorboardReporter(...) as tb:
            trainer = ...
            for _ in range(100):
                ...

    :param callbacks:
    :param save_dir:
    :param report_freq:
    :param report_params_freq: If n>0, reports parameter histograms every n iteration. if n==-1, every epoch.
    :param report_images_freq: If n>0, reports images every n iteration. if n==-1, every epoch.
    :param image_keys: keys for images.
    :param save_images: If True, saving images in addition to reporting
    """

    def __init__(self, callbacks: Union[Iterable[Callback], Callback],
                 save_dir=None,
                 report_freq: int = -1,
                 report_params_freq: int = 0,
                 report_images_freq: int = 0,
                 image_keys: Optional[Iterable[str]] = None,
                 save_images: bool = False):
        super(TensorboardReporter, self).__init__(TensorBoardWrapper(save_dir=save_dir, save_images=save_images),
                                                  callbacks, report_freq,
                                                  report_param_freq=report_params_freq,
                                                  report_image_freq=report_images_freq, image_keys=image_keys, )


class VisdomReporter(object):
    """
    Deprecated. Use TensorboardReporter instead.
    """

    def __init__(self, **kwargs):
        raise DeprecationWarning("VisdomReporter is no longer supported!")
