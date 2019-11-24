import functools
import time
from contextlib import contextmanager
from typing import Optional, Callable, Dict

import numpy as np
import torch

from homura.liblog import get_logger

logger = get_logger(__name__)


@contextmanager
def _syncronize(is_cuda: bool):
    if is_cuda:
        torch.cuda.synchronize()
    yield
    if is_cuda:
        torch.cuda.synchronize()


def timeit(func: Optional[Callable] = None,
           num_iters: Optional[int] = 100,
           warmup_iters: Optional[int] = None):
    """ A simple

    >>> @timeit(num_iters=100, warmup_iters=100)
    >>> def mm(a, b):
    >>>     return a @ b
    [homura.utils.benchmarks|2019-11-24 06:40:46|INFO] f requires 0.000021us per iteration
    """

    def _wrap(func):
        @functools.wraps(func)
        def _timeit(*args, **kwargs) -> Dict[str, float]:
            is_cuda = False
            for v in args:
                if torch.is_tensor(v) and v.is_cuda:
                    is_cuda = True
            for v in kwargs.values():
                if torch.is_tensor(v) and v.is_cuda:
                    is_cuda = True

            if is_cuda and warmup_iters is None:
                logger.warning("For benchmarking GPU computation, warmup is recommended.")

            if warmup_iters is not None:
                for _ in range(warmup_iters):
                    func(*args, **kwargs)

            times = [0] * num_iters
            with _syncronize(is_cuda):
                t0 = time.perf_counter()
                for i in range(num_iters):
                    t1 = time.perf_counter()
                    func(*args, **kwargs)
                    times[i] = time.perf_counter() - t1

            total_time = time.perf_counter() - t0
            logger.info(f"{func.__name__} requires {total_time / num_iters:3f}us per iteration")
            times = np.array(times)
            return {"total_time": total_time,
                    "mean": total_time / num_iters,
                    "median": np.median(times),
                    "min": np.min(times),
                    "max": np.max(times),
                    "std": np.std(times)}

        return _timeit

    return _wrap if func is None else _wrap(func)
