import functools
import statistics
import time
from contextlib import contextmanager
from typing import Callable

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


def timeit(func: Callable = None,
           num_iters: int = 100,
           warmup_iters: int = None):
    """ A simple timeit for GPU operations.

    >>> @timeit(num_iters=100, warmup_iters=100)
    >>> def mm(a, b):
    >>>     return a @ b
    >>> mm(a, b)
    [homura.utils.benchmarks|2019-11-24 06:40:46|INFO] f requires 0.000021us per iteration
    """

    def _wrap(func):
        @functools.wraps(func)
        def _timeit(*args, **kwargs) -> dict[str, float]:
            is_cuda = False
            for v in args:
                if isinstance(v, torch.Tensor) and v.is_cuda:
                    is_cuda = True
            for v in kwargs.values():
                if isinstance(v, torch.Tensor) and v.is_cuda:
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
            mean = statistics.mean(times)
            std = statistics.stdev(times)
            logger.info(f"{func.__name__} requires {mean:.4e}±{std:.4e} sec/iteration")
            return {"total_time": total_time,
                    "mean": total_time / num_iters,
                    "median": statistics.median(times),
                    "min": min(times),
                    "max": max(times),
                    "std": std}

        return _timeit

    return _wrap if func is None else _wrap(func)
