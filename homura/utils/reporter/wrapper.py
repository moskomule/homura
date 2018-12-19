import json
import numbers
import pathlib
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from matplotlib.figure import Figure
from torchvision.utils import make_grid

from homura.utils._miscs import get_git_hash
from homura.utils._vocabulary import *

DEFAULT_SAVE_DIR = "results"


def _dimension(x):
    if "numpy" in str(type(x)):
        dim = x.ndim
    elif "Tensor" in str(type(x)):
        # should be torch.**Tensor
        dim = x.dim()
    else:
        raise TypeError("Unknown Type!")

    return dim


def _to_numpy(x):
    if isinstance(x, numbers.Number):
        x = np.array([x])
    elif "Tensor" in str(type(x)):
        x = x.numpy()
    elif isinstance(x, Figure):
        x = Image.frombytes("RGB", x.canvas.get_width_height(), x.canvas.tostring_rgb())
        x = np.asanyarray(x)
    if not isinstance(x, np.ndarray):
        raise TypeError(f"Unknown type: {type(x)}")
    return x


class ReporterWrapper(object):
    def __init__(self, save_dir):
        """
        base class of Reporter
        >>> with ReporterWrapper("save_dir") as r:
        >>>     r.add_scalar(1, "loss", idx=0)
        # automatically save the results
        """
        self._container = defaultdict(list)
        save_dir = DEFAULT_SAVE_DIR if save_dir is None else save_dir
        self._save_dir = pathlib.Path(save_dir) / NOW
        self._filename = get_git_hash() + ".json"

    def add_scalar(self, x, name: str, idx: int):
        raise NotImplementedError

    def add_scalars(self, x: dict, name, idx: int):
        raise NotImplementedError

    def add_histogram(self, x, name: str, idx: int):
        raise NotImplementedError

    def add_text(self, x, name: str, idx: int):
        raise NotImplementedError

    def add_image(self, x, name: str, idx: int):
        raise NotImplementedError

    def add_images(self, x, name: str, idx: int):
        raise NotImplementedError

    def _register_data(self, x, name: str, idx: int):
        x = x if isinstance(x, str) else float(x)
        self._container[name].append((idx, x))

    def save(self):
        if self._save_dir:
            p = pathlib.Path(self._save_dir)
            p.mkdir(parents=True, exist_ok=True)
            with (p / self._filename).open("w") as f:
                json.dump(self._container, f)

    def close(self):
        self.save()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class TQDMWrapper(ReporterWrapper):
    def __init__(self, iterator, save_dir=None):
        """
        >>> with TQDMWrapper(range(100)) as tqrange:
        >>>     for i in tqrange:
        >>>         pass
        """
        from tqdm import tqdm

        super(TQDMWrapper, self).__init__(save_dir)
        self._tqdm = tqdm(iterator, ncols=80)
        self._size = len(iterator)

    def __iter__(self):
        for x in self._tqdm:
            yield x

    def __len__(self):
        return self._size

    def add_scalar(self, x, name: str, idx: int):
        self._register_data(x, name, idx)
        self._tqdm.set_postfix({name: x})

    def add_scalars(self, x: dict, name, idx: int):
        assert isinstance(x, dict)
        for k, v in x.items():
            self._register_data(v, k, idx)
        self._tqdm.set_postfix(x)

    def add_text(self, x, name: str, idx: int):
        pass

    def add_histogram(self, x, name: str, idx: int):
        pass

    def add_image(self, x, name: str, idx: int):
        pass

    def add_images(self, x, name: str, idx: int):
        pass


class VisdomWrapper(ReporterWrapper):
    def __init__(self, port=6006, save_dir=None):
        from visdom import Visdom

        super(VisdomWrapper, self).__init__(save_dir)
        self._viz = Visdom(port=port, env=NOW)
        self._lines = defaultdict()
        if not self._viz.check_connection():
            print(f"""
        Please launch visdom.server before calling VisdomWrapper.
        $python -m visdom.server -port {port}
        """)

    def add_scalar(self, x, name: str, idx: int, **kwargs):
        self.add_scalars({name: x}, name=name, idx=idx, **kwargs)

    def add_scalars(self, x: dict, name, idx: int, **kwargs):
        x = {k: _to_numpy(v) for k, v in x.items()}
        num_lines = len(x)
        is_new = self._lines.get(name) is None
        self._lines[name] = 1  # any non-None value
        for k, v in x.items():
            self._register_data(v, k, idx)
        opts = dict(title=name, legend=list(x.keys()))
        opts.update(**kwargs)
        X = np.column_stack([_to_numpy(idx) for _ in range(num_lines)])
        Y = np.column_stack([i for i in x.values()])
        if num_lines == 1:
            X, Y = X.reshape(-1), Y.reshape(-1)
        self._viz.line(X=X, Y=Y, update=None if is_new else "append", win=name, opts=opts)

    def add_histogram(self, x, name: str, idx: int, **kwargs):
        # todo
        raise NotImplementedError

    def add_text(self, x, name: str, idx: int):
        self._register_data(x, name, idx)
        self._viz.text(x)

    def add_image(self, x, name: str, idx: int):
        assert _dimension(x) == 3
        self._viz.image(self._normalize(x), opts=dict(title=name, caption=str(idx)))

    def add_images(self, x, name: str, idx: int):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if x.dim() == 3:
            x.unsqueeze_(0)
        self._viz.images(self._normalize(x), opts=dict(title=name, caption=str(idx)))

    def close(self):
        super(VisdomWrapper, self).close()
        self._viz.save([NOW])

    @staticmethod
    def _normalize(x):
        # normalize tensor values in (0, 1)
        _min, _max = x.min(), x.max()
        return (x - _min) / (_max - _min)


class TensorBoardWrapper(ReporterWrapper):
    def __init__(self, save_dir=None):
        from tensorboardX import SummaryWriter

        super(TensorBoardWrapper, self).__init__(save_dir)
        self._writer = SummaryWriter(log_dir=str(self._save_dir))

    def add_scalar(self, x, name: str, idx: int):
        self._register_data(x, name, idx)
        self._writer.add_scalar(name, x, idx)

    def add_scalars(self, x: dict, name, idx: int):
        for k, v in x.items():
            self._register_data(v, k, idx)
        self._writer.add_scalars(name, x, idx)

    def add_image(self, x, name: str, idx: int):
        assert _dimension(x) == 3
        self._writer.add_image(name, x, idx)

    def add_images(self, x, name: str, idx: int):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if x.dim() == 3:
            x.unsqueeze_(0)
        x = make_grid(x, normalize=True)
        self._writer.add_image(name, x, idx)

    def add_text(self, x, name: str, idx: int):
        self._register_data(x, name, idx)
        self._writer.add_text(name, x, idx)

    def add_histogram(self, x, name: str, idx: int):
        self._writer.add_histogram(name, x, idx, bins="sqrt")

    def close(self):
        super(TensorBoardWrapper, self).close()
        self._writer.close()
