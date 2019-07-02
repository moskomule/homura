# Backend classes of reporters

import json
import pathlib
import sys as python_sys
from abc import ABCMeta
from collections import defaultdict
from numbers import Number
from pathlib import Path
from typing import Union, Dict, Mapping, List

import numpy as np
import torch
from torchvision.utils import make_grid, save_image as _save_image

import homura
from homura.liblog import _set_tqdm_handler
from ._vocabulary import *
from .environment import get_git_hash

DEFAULT_SAVE_DIR = "results"
Vector = Union[Number, torch.Tensor, np.ndarray, List[Number]]


def _dimension(x: Vector):
    if isinstance(x, np.ndarray):
        dim = x.ndim
    elif torch.is_tensor(x):
        dim = x.dim()
    elif isinstance(x, List) and isinstance(x[0], Number):
        dim = 1
    elif isinstance(x, Number):
        dim = 0
    else:
        raise TypeError("Unknown Type!")

    return dim


def _num_elements(x: Vector):
    if isinstance(x, np.ndarray):
        elem = x.size
    elif torch.is_tensor(x):
        elem = x.nelement()
    elif isinstance(x, Number):
        elem = 1
    else:
        raise TypeError("Unknown Type!")
    return elem


def _to_numpy(x):
    if isinstance(x, Number):
        x = np.array([x])
    elif "Tensor" in str(type(x)):
        x = x.numpy()
    if not isinstance(x, np.ndarray):
        raise TypeError(f"Unknown type: {type(x)}")
    return x


def make_dir(path: str or Path) -> Path:
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path


def save_image(path: Path, x: torch.Tensor, name: str, idx: int) -> None:
    p = make_dir(path / "images" / str(idx))
    filename = p / f"{name}-{idx}.png"
    if not filename.exists():
        _save_image(x, filename)


def vector_to_dict(x: Union[Dict, Vector]):
    if isinstance(x, Mapping):
        x = {str(k): v for k, v in x.items()}
    else:
        if _dimension(x) == 1:
            x = {str(i): v for i, v in enumerate(x)}
        else:
            raise TypeError(f"Unknown type! {type(x)}: {x}")
    return x


class _WrapperBase(metaclass=ABCMeta):
    def __init__(self, save_dir):
        self._container = defaultdict(list)
        self._container["args"] = " ".join(python_sys.argv)

        save_dir = DEFAULT_SAVE_DIR if save_dir is None else save_dir
        postfix = ""
        if len(get_git_hash()) > 0:
            postfix = "-" + get_git_hash()
        self._save_dir = pathlib.Path(save_dir) / (BASIC_DIR_NAME + postfix)
        self._filename = NOW + ".json"
        self.logger = homura.liblog.get_logger(__name__)

    def add_scalar(self, x: Vector, name: str, idx: int):
        raise NotImplementedError

    def add_scalars(self, x: Dict[str, Vector], name, idx: int):
        raise NotImplementedError

    def add_histogram(self, x: torch.Tensor, name: str, idx: int):
        raise NotImplementedError

    def add_text(self, x: str, name: str, idx: int):
        raise NotImplementedError

    def add_image(self, x: torch.Tensor, name: str, idx: int):
        raise NotImplementedError

    def add_images(self, x: torch.Tensor, name: str, idx: int):
        raise NotImplementedError

    def _register_data(self, x: Union[Vector, str], name: str, idx: int):
        x = x if isinstance(x, str) else float(x)
        self._container[name].append((idx, x))

    def save(self):
        if self._save_dir is not None and not (self._save_dir / self._filename).exists():
            p = make_dir(self._save_dir)
            with (p / self._filename).open("w") as f:
                json.dump(self._container, f)

    def close(self):
        self.save()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class _NoOpWrapper(_WrapperBase):
    """ NoOp for distributed training
    """

    def __init__(self, *args, **kwargs):
        pass

    def save(self):
        pass

    def add_text(self, x: str, name: str, idx: int):
        pass

    def add_histogram(self, x: torch.Tensor, name: str, idx: int):
        pass

    def add_scalar(self, x: Vector, name: str, idx: int):
        pass

    def add_scalars(self, x: Dict[str, Vector], name, idx: int):
        pass

    def add_image(self, x: torch.Tensor, name: str, idx: int):
        pass

    def add_images(self, x: torch.Tensor, name: str, idx: int):
        pass


class TQDMWrapper(_WrapperBase):
    def __init__(self, iterator, save_dir=None):
        from tqdm import tqdm

        super(TQDMWrapper, self).__init__(save_dir)
        self.tqdm = tqdm(iterator, ncols=80)
        self._size = len(iterator)
        _set_tqdm_handler()

    def __iter__(self):
        for x in self.tqdm:
            yield x

    def __len__(self):
        return self._size

    def add_scalar(self, x, name: str, idx: int):
        self._register_data(x, name, idx)
        self.tqdm.set_postfix({name: x})

    def add_scalars(self, x: dict, name: str, idx: int):
        x = vector_to_dict(x)
        for k, v in x.items():
            self._register_data(v, k, idx)
        self.tqdm.set_postfix(x)

    def add_text(self, x, name: str, idx: int):
        self._register_data(x, name, idx)
        self.logger.info(f"[{idx:>10}]{name}={x}")

    def add_histogram(self, x, name: str, idx: int):
        self.logger.warning(f"{self.__class__.__name__} cannot report histogram!")

    def add_image(self, x, name: str, idx: int):
        save_image(self._save_dir, x, name, idx)

    def add_images(self, x, name: str, idx: int):
        save_image(self._save_dir, x, name, idx)


class LoggerWrapper(_WrapperBase):
    def __init__(self, logger=None, save_dir=None, save_log: bool = False):
        super(LoggerWrapper, self).__init__(save_dir)
        from homura.liblog import get_logger, set_file_handler

        self.logger = get_logger("homura.reporter") if logger is None else logger
        if save_log:
            set_file_handler(self._save_dir / "log.txt")

    def add_scalar(self, x, name: str, idx: int):
        self._register_data(x, name, idx)
        self.logger.info(f"[{idx:>7}] {name:>30} {x:.4f}")

    def add_scalars(self, x: Union[Dict, Vector], name, idx: int):
        x = vector_to_dict(x)
        name = name or ""  # if name is None, then ""
        for k, v in x.items():
            k = name + str(k)
            self._register_data(v, k, idx)
            self.logger.info(f"[{idx:>7}] {k:>30} {v:.4f}")

    def add_text(self, x: str, name: str, idx: int):
        self._register_data(x, name, idx)
        self.logger.info(f"[{idx:>7}] {name:>30} {x}")

    def add_histogram(self, x, name: str, idx: int):
        self.logger.warning(f"{self.__class__.__name__} cannot report histogram!")

    def add_image(self, x, name: str, idx: int):
        save_image(self._save_dir, x, name, idx)

    def add_images(self, x, name: str, idx: int):
        save_image(self._save_dir, x, name, idx)


class TensorBoardWrapper(_WrapperBase):

    def __new__(cls, *args, **kwargs):
        if homura.get_global_rank() > 0:
            return _NoOpWrapper()
        else:
            return object.__new__(cls)

    def __init__(self, save_dir=None, save_images=False):
        from torch.utils.tensorboard import SummaryWriter

        super(TensorBoardWrapper, self).__init__(save_dir)
        self._writer = SummaryWriter(log_dir=str(self._save_dir))
        self._save_image = save_images

    def add_scalar(self, x: Vector, name: str, idx: int):
        self._register_data(x, name, idx)
        self._writer.add_scalar(name, x, idx)

    def add_scalars(self, x: Union[Dict, Vector], name: str, idx: int):
        x = vector_to_dict(x)
        for k, v in x.items():
            self._register_data(v, k, idx)
        self._writer.add_scalars(name, x, idx)

    def add_image(self, x: Vector, name: str, idx: int):
        assert _dimension(x) == 3
        self._writer.add_image(name, x, idx)
        if self._save_image:
            save_image(self._save_dir, x, name, idx)

    def add_images(self, x: Vector, name: str, idx: int):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if x.dim() == 3:
            x.unsqueeze_(0)
        x = make_grid(x, normalize=True)
        self._writer.add_image(name, x, idx)
        if self._save_image:
            save_image(self._save_dir, x, name, idx)

    def add_text(self, x: str, name: str, idx: int):
        self._register_data(x, name, idx)
        self._writer.add_text(name, x, idx)

    def add_histogram(self, x: torch.Tensor, name: str, idx: int):
        self._writer.add_histogram(name, x, idx, bins="sqrt")

    def close(self):
        super(TensorBoardWrapper, self).close()
        self._writer.close()
