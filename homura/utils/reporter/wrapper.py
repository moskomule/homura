import json
import pathlib
from abc import ABCMeta
from collections import defaultdict
from numbers import Number
from pathlib import Path
from typing import Union, Dict

import numpy as np
import torch
from PIL import Image
from matplotlib.figure import Figure
from torchvision.utils import make_grid, save_image as _save_image

import homura
from homura.utils._miscs import get_git_hash
from homura.utils._vocabulary import *

DEFAULT_SAVE_DIR = "results"
Vector = Union[Number, torch.Tensor, np.ndarray]


def _dimension(x: Vector):
    if isinstance(x, np.ndarray):
        dim = x.ndim
    elif torch.is_tensor(x):
        dim = x.dim()
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
    elif isinstance(x, Figure):
        x = Image.frombytes("RGB", x.canvas.get_width_height(), x.canvas.tostring_rgb())
        x = np.asanyarray(x)
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


class _WrapperBase(metaclass=ABCMeta):
    def __init__(self, save_dir):
        self._container = defaultdict(list)
        save_dir = DEFAULT_SAVE_DIR if save_dir is None else save_dir
        self._save_dir = pathlib.Path(save_dir) / (NOW + "-" + get_git_hash())
        self._filename = NOW + ".json"

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
        if self._save_dir is not None:
            p = make_dir(self._save_dir)
            with (p / self._filename).open("w") as f:
                json.dump(self._container, f)

    def close(self):
        self.save()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class TQDMWrapper(_WrapperBase):
    def __init__(self, iterator, save_dir=None):
        from tqdm import tqdm

        super(TQDMWrapper, self).__init__(save_dir)
        self.tqdm = tqdm(iterator, ncols=80)
        self._size = len(iterator)
        self.logger = homura.liblog.get_logger(__name__)

    def __iter__(self):
        for x in self.tqdm:
            yield x

    def __len__(self):
        return self._size

    def add_scalar(self, x, name: str, idx: int):
        self._register_data(x, name, idx)
        self.tqdm.set_postfix({name: x})

    def add_scalars(self, x: dict, name: str, idx: int):
        assert isinstance(x, dict)
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
    def __init__(self, logger=None, save_dir=None):
        super(LoggerWrapper, self).__init__(save_dir)
        from homura.liblog import get_logger, set_verb_level

        self.logger = get_logger(self.__class__.__name__) if logger is None else logger
        set_verb_level("info")

    def add_scalar(self, x, name: str, idx: int):
        self._register_data(x, name, idx)
        self.logger.info(f"[{idx:>10}]{name}={x}")

    def add_scalars(self, x: dict, name, idx: int):
        assert isinstance(x, dict)
        for k, v in x.items():
            self._register_data(v, k, idx)
            self.logger.info(f"[{idx:>10}]{k}={v}")

    def add_text(self, x: str, name: str, idx: int):
        self._register_data(x, name, idx)
        self.logger.info(f"[{idx:>10}]{name}={x}")

    def add_histogram(self, x, name: str, idx: int):
        self.logger.warning(f"{self.__class__.__name__} cannot report histogram!")

    def add_image(self, x, name: str, idx: int):
        save_image(self._save_dir, x, name, idx)

    def add_images(self, x, name: str, idx: int):
        save_image(self._save_dir, x, name, idx)


class TensorBoardWrapper(_WrapperBase):
    def __init__(self, save_dir=None, save_images=False):
        if homura.is_tensorboard_available:
            from tensorboardX import SummaryWriter
        else:
            raise ImportError("To use TensorboardWrapper, tensorboardX is needed!")

        super(TensorBoardWrapper, self).__init__(save_dir)
        self._writer = SummaryWriter(log_dir=str(self._save_dir))
        self._save_image = save_images

    def add_scalar(self, x: Vector, name: str, idx: int):
        self._register_data(x, name, idx)
        self._writer.add_scalar(name, x, idx)

    def add_scalars(self, x: Dict[str, Vector], name: str, idx: int):
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
