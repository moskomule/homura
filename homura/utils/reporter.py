import pathlib
from collections import defaultdict
import json
import numbers

import numpy as np
from ._miscs import get_git_hash
from ._vocabulary import V


class Reporter(object):
    def __init__(self, save_dir):
        """
        base class of Reporter
        >>> with Reporter("save_dir") as r:
        >>>     r.add_scalar(1, "loss", idx=0)
        # automatically save the results
        """
        self._container = defaultdict(list)
        self._save_dir = save_dir
        self._filename = V.NOW + get_git_hash() + ".json"

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

    @staticmethod
    def _tensor_type_check(x):
        if "numpy" in str(type(x)):
            dim = x.ndim
        elif "Tensor" in str(type(x)):
            # should be torch.**Tensor
            x = x.cpu()
            dim = x.dim()
        else:
            raise TypeError("Unknown Type!")

        return x, dim


class ReporterList(Reporter):
    def __init__(self, *reporters):
        super(ReporterList, self).__init__(None)
        for r in reporters:
            assert isinstance(r, Reporter)
        self.reporters = reporters

    def add_scalar(self, x, name: str, idx: int):
        for r in self.reporters:
            r.add_scalar(x, name, idx)

    def add_scalars(self, x: dict, name, idx: int):
        for r in self.reporters:
            r.add_scalars(x, name, idx)

    def add_histogram(self, x, name: str, idx: int):
        for r in self.reporters:
            r.add_histogram(x, name, idx)

    def add_text(self, x, name: str, idx: int):
        for r in self.reporters:
            r.add_text(x, name, idx)

    def add_image(self, x, name: str, idx: int):
        for r in self.reporters:
            r.add_image(x, name, idx)

    def add_images(self, x, name: str, idx: int):
        for r in self.reporters:
            r.add_images(x, name, idx)

    def close(self):
        for r in self.reporters:
            r.close()

    def __iter__(self):
        for x in self.reporters:
            yield x


class TQDMReporter(Reporter):
    def __init__(self, iterable, save_dir=None):
        """
        >>> with TQDMReporter(range(100)) as tqrange:
        >>>     for i in tqrange:
        >>>         pass
        """
        from tqdm import tqdm

        super(TQDMReporter, self).__init__(save_dir)
        self._tqdm = tqdm(iterable, ncols=80)
        self._size = len(iterable)

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


class VisdomReporter(Reporter):
    def __init__(self, port=6006, save_dir=None):
        from visdom import Visdom

        super(VisdomReporter, self).__init__(save_dir)
        self._viz = Visdom(port=port, env=V.NOW)
        self._lines = defaultdict()
        if not self._viz.check_connection():
            print(f"""
        Please launch visdom.server before calling VisdomReporter.
        $python -m visdom.server -port {port}
        """)

    def add_scalar(self, x, name: str, idx: int, **kwargs):
        self.add_scalars({name: x}, name=name, idx=idx, **kwargs)

    def add_scalars(self, x: dict, name, idx: int, **kwargs):
        x = {k: self._to_numpy(v) for k, v in x.items()}
        num_lines = len(x)
        is_new = self._lines.get(name) is None
        self._lines[name] = 1  # any non-None value
        for k, v in x.items():
            self._register_data(v, k, idx)
        opts = dict(title=name, legend=list(x.keys()))
        opts.update(**kwargs)
        X = np.column_stack([self._to_numpy(idx) for _ in range(num_lines)])
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
        x, dim = self._tensor_type_check(x)
        assert dim == 3
        self._viz.image(self._normalize(x), opts=dict(title=name, caption=str(idx)))

    def add_images(self, x, name: str, idx: int):
        x, dim = self._tensor_type_check(x)
        assert dim == 4
        self._viz.images(self._normalize(x), opts=dict(title=name, caption=str(idx)))

    def _to_numpy(self, x):
        if isinstance(x, numbers.Number):
            x = np.array([x])
        elif "Tensor" in str(type(x)):
            x = x.numpy()
        return x

    def close(self):
        super(VisdomReporter, self).close()
        self._viz.save([V.NOW])

    @staticmethod
    def _normalize(x):
        # normalize tensor values in (0, 1)
        _min, _max = x.min(), x.max()
        return (x - _min) / (_max - _min)


class TensorBoardReporter(Reporter):
    def __init__(self, save_dir=None):
        from tensorboardX import SummaryWriter

        super(TensorBoardReporter, self).__init__(save_dir)
        self._writer = SummaryWriter(log_dir=save_dir)

    def add_scalar(self, x, name: str, idx: int):
        self._register_data(x, name, idx)
        self._writer.add_scalar(name, x, idx)

    def add_scalars(self, x: dict, name, idx: int):
        for k, v in x.items():
            self._register_data(v, k, idx)
        self._writer.add_scalars(name, x, idx)

    def add_image(self, x, name: str, idx: int):
        x, dim = self._tensor_type_check(x)
        assert dim == 3
        self._writer.add_image(name, x, idx)

    def add_images(self, x, name: str, idx: int):
        pass

    def add_text(self, x, name: str, idx: int):
        self._register_data(x, name, idx)
        self._writer.add_text(name, x, idx)

    def add_histogram(self, x, name: str, idx: int):
        self._writer.add_histogram(name, x, idx, bins="sqrt")

    def close(self):
        super(TensorBoardReporter, self).close()
        self._writer.close()
