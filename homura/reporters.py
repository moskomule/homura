from __future__ import annotations

import warnings
from collections import defaultdict
from numbers import Number
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
from torch import distributed

from homura import get_args, if_is_master, is_distributed, is_master, liblog

__all__ = ["ReporterList", "TensorboardReporter", "TQDMReporter"]

Value = torch.Tensor or Number or dict[str, torch.Tensor or Number]


class _ReporterBase(object):

    def flush(self):
        pass

    def add_text(self,
                 key: str,
                 value: str,
                 step: int = None
                 ) -> None:
        pass

    def add_scalar(self,
                   key: str,
                   value: Number or torch.Tensor,
                   step: int = None
                   ) -> None:
        pass

    def add_scalars(self,
                    key: str,
                    value: dict[str, Number or torch.Tensor],
                    step: int = None
                    ) -> None:
        pass


class TQDMReporter(_ReporterBase):
    def __init__(self,
                 ncols: int = 80
                 ) -> None:
        self.writer = None
        self._ncols = ncols
        self._temporal_memory = {}

        liblog.set_tqdm_handler()
        liblog.set_tqdm_stdout_stderr()

    def set_iterator(self,
                     iterator: Iterator
                     ) -> None:
        if is_master():
            self.writer = liblog.tqdm(iterator, ncols=self._ncols)
        else:
            self.writer = iterator

    def __iter__(self):
        for i in self.writer:
            yield i

    def __len__(self
                ) -> int:
        return len(self.writer)

    @if_is_master
    def flush(self):
        postfix = {key: value
                   for key, (value, _) in self._temporal_memory.items()
                   if isinstance(value, Number)}
        self.writer.set_postfix(postfix)

        if len(postfix) != len(self._temporal_memory):
            for k, v in {key: value
                         for key, (value, _) in self._temporal_memory.items()
                         if not isinstance(value, Number)}.items():
                self.add_text(k, v)
        # clear temporal memory
        self._temporal_memory = {}

    @if_is_master
    def add_text(self,
                 key: str,
                 value: str,
                 step: int = None
                 ) -> None:
        self.writer.write(value)

    @if_is_master
    def add_scalar(self,
                   key: str,
                   value: Number or torch.Tensor,
                   step: int = None
                   ) -> None:
        if isinstance(value, torch.Tensor):
            value = value.item()
        self._temporal_memory[key] = (value, step)

    @if_is_master
    def add_scalars(self,
                    key: str,
                    value: dict[str, Number or torch.Tensor],
                    step: int = None
                    ) -> None:
        self._temporal_memory[key] = (value, step)


class TensorboardReporter(_ReporterBase):
    def __init__(self,
                 save_dir: str = None
                 ) -> None:
        if is_master():
            from torch.utils import tensorboard
            self._save_dir = Path(save_dir or ".")
            self._save_dir.mkdir(exist_ok=True, parents=True)
            self.writer = tensorboard.SummaryWriter(save_dir)
            self.writer.add_text("exec", ' '.join(get_args()))

    @if_is_master
    def add_text(self,
                 key: str,
                 value: str,
                 step: int = None
                 ) -> None:
        self.writer.add_text(key, value, step)

    @if_is_master
    def add_audio(self,
                  key: str,
                  audio: torch.Tensor,
                  step: int = None
                  ) -> None:
        if audio.ndim != 2 or audio.size(0) != 1:
            raise RuntimeError(f"Shape of audio tensor is expected to be [1, L], but got {audio.shape}")
        self.writer.add_audio(key, audio, step)

    @if_is_master
    def add_histogram(self,
                      key: str,
                      values: torch.Tensor,
                      step: int,
                      bins: str = 'tensorflow'
                      ) -> None:
        self.writer.add_histogram(key, values, step, bins=bins)

    @if_is_master
    def add_image(self,
                  key: str,
                  image: torch.Tensor,
                  step: int = None
                  ) -> None:
        dim = image.dim()
        if dim == 3:
            self.writer.add_image(key, image, step)
        elif dim == 4:
            self.writer.add_images(key, image, step)
        else:
            raise ValueError(f"Dimension of image tensor is expected to be 3 or 4, but got {dim}")

    @if_is_master
    def add_scalar(self,
                   key: str,
                   value: Any,
                   step: int = None
                   ) -> None:
        self.writer.add_scalar(key, value, step)

    @if_is_master
    def add_scalars(self,
                    key: str,
                    value: dict[str, Any],
                    step: int = None
                    ) -> None:
        self.writer.add_scalars(key, value, step)

    @if_is_master
    def add_figure(self,
                   key: str,
                   figure: "matplotlib.pyplot.figure",
                   step: int = None
                   ) -> None:
        self.writer.add_figure(key, figure, step)


class _Accumulator(object):
    # for accumulation and sync
    def __init__(self,
                 key: str,
                 reduction: str or Callable,
                 no_sync: bool
                 ) -> None:
        self._key = key
        if isinstance(reduction, str) and reduction not in {'sum', 'average'}:
            raise ValueError(f"reduction is expected to be 'sum' or 'average', but got {reduction}.")

        self._reduction = reduction
        self._sync = not no_sync and is_distributed()
        self._total_size: int = 0

        self._memory: list[Any] = []

    def set_batch_size(self,
                       batch_size: int
                       ) -> None:
        if self._sync:
            _batch_size = torch.empty(1, dtype=torch.int,
                                      device=torch.device(torch.cuda.current_device())
                                      ).fill_(batch_size)
            distributed.all_reduce(_batch_size, op=distributed.ReduceOp.SUM)
            batch_size = _batch_size.item()
        self._total_size += batch_size

    def __call__(self,
                 value: Value
                 ) -> _Accumulator:
        # value is extpected to be
        # 1. Number
        # 2. Tensor
        # 3. dict[str, Number or Tensor]
        value = self._process_tensor(value)

        if isinstance(value, dict):
            value = {k: self._process_tensor(v) for k, v in value.items()}

        self._memory.append(value)
        return self

    def _process_tensor(self,
                        value: Value
                        ) -> Value:
        if torch.is_tensor(value):
            if self._sync:
                distributed.all_reduce(value, op=distributed.ReduceOp.SUM)
            value = value.cpu()
        return value

    def _reduce(self,
                values: list[Value]
                ) -> Value:
        if self._reduction == 'sum':
            return sum(values)
        elif self._reduction == 'average':
            return sum(values) / self._total_size
        else:
            return self._reduction(values)

    def accumulate(self
                   ) -> Value:
        # called after iteration

        if isinstance(self._memory[0], dict):
            # _memory is [{k: v}, {k: v}, ....]
            return {k: self._reduce([d[k] for d in self._memory])
                    for k in self._memory[0].keys()}

        return self._reduce(self._memory)


class _History(dict):
    # Dictionary that can be access via () and []
    def __init__(self, history_dict: dict[str, list[float]]) -> None:
        super().__init__(history_dict)

    def __getitem__(self,
                    item: str
                    ) -> list[float]:
        return super().__getitem__(item)

    __call__ = __getitem__
    __getitem__ = __getitem__


class ReporterList(object):
    """ ReporterList is expected to be used in TrainerBase
    """

    # _persistent_hist tracks scalar values
    _persistent_hist: dict[str, list[Value]] = defaultdict(list)

    def __init__(self,
                 reporters: list[_ReporterBase]
                 ) -> None:
        self.reporters = reporters
        # _epoch_hist clears after each epoch
        self._batch_size: int = None
        self._epoch_hist: dict[str, _Accumulator] = {}

    def set_batch_size(self,
                       batch_size: int
                       ) -> None:
        # intended to be used in trainer
        self._batch_size = batch_size

    def add_value(self,
                  key: str,
                  value: Value,
                  *,
                  is_averaged: bool = True,
                  reduction: str or Callable[[Value, ...], Value] = 'average',
                  no_sync: bool = False,
                  ) -> None:
        """ Add value(s) to reporter ::

            def iteration(self: TrainerBase, data: Tuple[Tensor, ...]):

                self.reporter.add_value('loss', loss.detach())
                self.reporter.add_value('miou', confusion_matrix(output, target), reduction=cm_to_miou)

        :param key: Unique key to track value
        :param value: Value
        :param is_averaged: If value is averaged
        :param reduction: Method of reduction after epoch, 'average', 'sum' or function of list[Value] -> Value
        :param no_sync: If not sync in distributed setting
        :return:
        """

        if is_averaged:
            value *= self._batch_size

        if self._epoch_hist.get(key) is None:
            self._epoch_hist[key] = _Accumulator(key, reduction, no_sync)(value)
        else:
            self._epoch_hist[key](value)

        self._epoch_hist[key].set_batch_size(self._batch_size)

    __call__ = add_value
    add = add_value

    def _add_backend(self,
                     something: str,
                     *args,
                     **kwargs):
        used = False
        for rep in self.reporters:
            if hasattr(rep, something):
                used = True
                getattr(rep, something)(*args, **kwargs)
        if not used:
            warnings.warn(f"None of reporters has attribute: {something}")

    @if_is_master
    def add_figure(self,
                   key: str,
                   figure: "matplotlib.pyplot.figure",
                   step: int = None
                   ) -> None:
        """ Report Figure of matplotlib.pyplot
        """
        self._add_backend("add_figure", key, figure, step)

    @if_is_master
    def add_histogram(self,
                      key: str,
                      value: torch.Tensor,
                      step: int = None,
                      bins: str = "tensorflow"
                      ) -> None:
        """ Report histogram of a given tensor
        """
        self._add_backend("add_histogram", key, value, step, bins=bins)

    @if_is_master
    def add_image(self,
                  key: str,
                  image: torch.Tensor,
                  step: int = None,
                  normalize: bool = False
                  ) -> None:
        """ Report a single image or a batch of images
        """
        if normalize:
            image = self._norm_image(image)
        self._add_backend("add_image", key, image, step)

    @staticmethod
    @torch.no_grad()
    def _norm_image(img: torch.Tensor
                    ) -> torch.Tensor:
        img = img.clone()
        min, max = img.min().item(), img.max().item()
        img.clamp_(min=min, max=max)
        img.add_(-min).div_(max - min + 1e-5)
        return img

    @if_is_master
    def add_text(self,
                 key: str,
                 text: str,
                 step: int = None
                 ) -> None:
        """ Report text
        """
        self._add_backend("add_text", key, text, step)

    def report(self,
               step: int = None,
               mode: str = ""
               ) -> None:
        # intended to be called after epoch
        if len(self._epoch_hist) == 0:
            # to avoid report repeatedly in a single epoch
            return

        temporal_memory = {}
        for k, v in self._epoch_hist.items():
            # accumulate stored values during an epoch
            key = f"{k}/{mode}" if len(mode) > 0 else k
            accumulated = v.accumulate()
            accumulated = (accumulated
                           if isinstance(accumulated, (Number, dict)) or isinstance(accumulated, torch.Tensor)
                           else None)
            self._persistent_hist[key].append(accumulated)
            temporal_memory[key] = accumulated

        for k, v in temporal_memory.items():
            if isinstance(v, torch.Tensor):
                if v.nelement() == 1:
                    for rep in self.reporters:
                        rep.add_scalar(k, v, step)
                else:
                    for rep in self.reporters:
                        rep.add_scalars(k, {str(i): vv for i, vv in enumerate(v.tolist())}, step)
            elif isinstance(v, Number):
                for rep in self.reporters:
                    rep.add_scalar(k, v, step)
            else:
                for rep in self.reporters:
                    rep.add_scalars(k, v, step)

        # cleanup
        for rep in self.reporters:
            rep.flush()
        self._clear_epoch_hist()

    @property
    def history(self
                ) -> _History:
        return _History(self._persistent_hist)

    def _clear_epoch_hist(self
                          ) -> None:
        self._epoch_hist = {}

    def exit(self
             ) -> None:
        # expected to be used in TrainerBase.exit
        ReporterList._persistent_hist = defaultdict(list)
