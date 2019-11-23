import json
from numbers import Number
from pathlib import Path
from typing import Mapping, Iterable, Optional, List

import torch
import tqdm
from torchvision.utils import save_image as _save_image

from homura import liblog
from homura.utils import get_global_rank
from homura.utils._vocabulary import *
from .base import Callback, CallbackList
from .metrics import MetricCallback

Callbacks = Callback or Iterable[Callback]
Vector = torch.Tensor or List[Number] or Number


class _ReporterBase(Callback):
    # Actual base class for reporters, but users do not inherit class

    def __init__(self, *args, **kwargs):
        self.callbacks = None

    def register_callbacks(self,
                           callbacks: CallbackList):
        self.callbacks = callbacks

    def before_all(self,
                   data: Mapping) -> Mapping:
        if self.callbacks is None:
            raise RuntimeError("`callbacks` are not registered.")
        return self.callbacks.before_all(data)

    def before_epoch(self,
                     data: Mapping) -> Mapping:
        return self.callbacks.before_epoch(data)

    def before_iteration(self,
                         data: Mapping) -> Mapping:
        return self.callbacks.before_iteration(data)

    def after_all(self,
                  data: Mapping) -> Mapping:
        return self.callbacks.after_all(data)

    def after_epoch(self,
                    data: Mapping) -> Mapping:
        return self.callbacks.after_epoch(data)

    def after_iteration(self,
                        data: Mapping) -> Mapping:
        return self.callbacks.after_iteration(data)

    def close(self):
        self.callbacks.close()


class Reporter(_ReporterBase):
    """ Virtual base class of reporters.
    If `global_rank>0`, i.e., not master node in distributed learning, `_ReporterBase` is used.
    """

    master_only = True

    def __new__(cls,
                *args,
                **kwargs):
        if get_global_rank() > 0:
            return _ReporterBase(*args, **kwargs)
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        super(Reporter, self).__init__(*args, **kwargs)

    @staticmethod
    def _is_scalar(v: Vector) -> bool:
        if torch.is_tensor(v):
            return v.numel() == 1
        return isinstance(v, Number)

    @staticmethod
    def _is_vector(v: Vector) -> bool:
        return (torch.is_tensor(v) and v.dim() == 1) or all([Reporter._is_scalar(e) for e in v])

    @staticmethod
    def _is_images(t: torch.Tensor) -> bool:
        return (t.dim() == 3) or (t.dim() == 4 and t.size(1) in (1, 3))


class TQDMReporter(Reporter):
    """ Reporter with TQDM

    :param iterator: iterator to be wrapped.

    >>> t_reporter = TQDMReporter(range(100))
    >>> for ep in t_reporter:
    >>>     pass
    """

    def __init__(self,
                 iterator: Iterable):

        super(TQDMReporter, self).__init__()
        self.writer = tqdm.tqdm(iterator, ncols=80)
        self._length = len(iterator)
        liblog._set_tqdm_handler()

    def __iter__(self):
        for i in self.writer:
            yield i

    def __len__(self):
        return self._length

    def add_text(self,
                 text: str):
        self.writer.write(text)

    def after_epoch(self,
                    data: Mapping):
        reportable = {}
        results = super(TQDMReporter, self).after_iteration(data)
        for k, v in results.items():
            if self._is_scalar(v):
                reportable[k] = v.item() if torch.is_tensor(v) else v
            elif isinstance(v, dict):
                reportable.update(v)
        self.writer.set_postfix(reportable)


class TensorboardReporter(Reporter):
    """ Reporter with Tensorboard

    :param save_dir: directory where the log is saved
    :report_freq: Frequency of reporting in iteration. If `None`, reported by epoch.
    """

    def __init__(self,
                 save_dir: Optional[str or Path],
                 report_freq: Optional[int] = None,
                 is_global_step_epoch: bool = True):
        super(TensorboardReporter, self).__init__()
        from torch.utils import tensorboard

        Path(save_dir).mkdir(exist_ok=True, parents=True)
        self.writer = tensorboard.SummaryWriter(save_dir)
        self._report_freq = report_freq
        self._use_epoch = is_global_step_epoch

    def after_iteration(self,
                        data: Mapping):
        results = super(TensorboardReporter, self).after_iteration(data)
        global_step = data[EPOCH if self._use_epoch else ITERATION]
        for k, v in results.items():
            if self._report_freq is not None and data[ITERATION] % self._report_freq == 0:
                self._report_values(k, v, global_step)
            elif self._is_images(v):
                self.writer.add_images(k, v, global_step)

    def after_epoch(self,
                    data: Mapping):
        results = super(TensorboardReporter, self).after_iteration(data)
        global_step = data[EPOCH if self._use_epoch else ITERATION]
        if self._report_freq is None:
            for k, v in results.items():
                self._report_values(k, v, global_step)

    def _report_values(self,
                       k: str,
                       v: Vector or dict,
                       global_step: int):

        if self._is_scalar(v):
            self.writer.add_scalar(k, v, global_step)
        elif isinstance(v, dict):
            self.writer.add_scalars(k, v, global_step)
        elif self._is_vector(v):
            self.writer.add_scalars(k, {str(i): e for i, e in enumerate(v)}, global_step)


class IOReporter(Reporter):
    """ Reporter based on IO, i.e., save json files for scalars and image files for images.
    """

    def __init__(self,
                 save_dir: Optional[str or Path]):
        super(IOReporter, self).__init__()
        Path(save_dir).mkdir(exist_ok=True, parents=True)
        self.save_dir = Path(save_dir)

    def after_iteration(self,
                        data: Mapping):
        # save image
        results = super(IOReporter, self).after_iteration(data)
        for k, v in results.items():
            if self._is_images(v):
                self.save_image(self.save_dir, v, k, data[EPOCH])

    def close(self):
        # save text
        history = {}
        if hasattr(self.callbacks, "callbacks"):
            for c in self.callbacks.callbacks:
                if isinstance(c, MetricCallback):
                    history[c.metric_name] = c.history
            with (self.save_dir / "results.json").open('w') as f:
                json.dump(history, f)

    @staticmethod
    def save_image(path: Path,
                   img: torch.Tensor,
                   name: str,
                   idx: int) -> None:
        (path / "images").mkdir(exist_ok=True, parents=True)
        filename = path / "images" / f"{name}-{idx}.png"
        _save_image(img, filename)


class CallImage(Callback):
    """ Fetch image from `data`. If want to report image by epoch, set `report_freq=None` (default)
    """

    master_only = True

    def __init__(self,
                 key: str,
                 report_freq: Optional[int] = None):
        self.key = key
        self.report_freq = report_freq

    def after_iteration(self,
                        data: Mapping):
        if data[ITERATION] == 0:
            if data.get(self.key) is None:
                raise RuntimeError(f"key for image `{self.key}` is not found in `data`")
        if self.report_freq is None:
            # epochwise, because `data` of `after_epoch` does not have images
            if data[ITERATION] % data[ITER_PER_EPOCH] != data[ITER_PER_EPOCH] - 1:
                return
        else:
            if data[ITERATION] % self.report_freq != 0:
                return

        return {self.key: data[self.key]}
