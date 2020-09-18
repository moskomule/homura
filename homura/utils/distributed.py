""" Helper functions to make distributed training easy
"""

import builtins
import importlib.util
import os as python_os
from functools import wraps
from typing import Callable, Optional

from torch import distributed
from torch.cuda import device_count

from homura.liblog import get_logger
from .environment import get_args, get_environ

logger = get_logger("homura.distributed")
# IS_DISTRIBUTED is used to handle horovod
IS_DISTRIBUTED_HOROVOD = False


def is_horovod_available() -> bool:
    return importlib.util.find_spec("horovod") is not None


def is_distributed_available() -> bool:
    return distributed.is_available() or is_horovod_available()


def is_distributed() -> bool:
    """ Check if the process is distributed by checking the world size is larger than 1.
    """

    return get_world_size() > 1


def get_local_rank() -> int:
    """ Get the local rank of the process, i.e., the process number of the node.
    """

    if IS_DISTRIBUTED_HOROVOD:
        import horovod.torch as hvd

        return hvd.local_rank()
    else:
        return int(get_environ('LOCAL_RANK', 0))


def get_global_rank() -> int:
    """ Get the global rank of the process. 0 if the process is the master.
    """

    if IS_DISTRIBUTED_HOROVOD:
        import horovod.torch as hvd

        return hvd.rank()
    else:
        return int(get_environ('RANK', 0))


def is_master() -> bool:
    return get_global_rank() == 0


def get_num_nodes() -> int:
    """ Get the number of nodes. Note that this function assumes all nodes have the same number of processes.
    """

    if not is_distributed():
        return 1
    else:
        return get_world_size() // device_count()


def get_world_size() -> int:
    """ Get the world size, i.e., the total number of processes.
    """

    if IS_DISTRIBUTED_HOROVOD:
        import horovod.torch as hvd

        return hvd.size()
    else:
        return int(python_os.environ.get("WORLD_SIZE", 1))


def init_distributed(use_horovod: bool = False,
                     backend: Optional[str] = None,
                     init_method: Optional[str] = None,
                     warning: bool = True):
    """ Simple initializer for distributed training.

    :param use_horovod: If use horovod as distributed backend
    :param backend: backend of torch.distributed.init_process_group
    :param init_method: init_method of torch.distributed.init_process_group
    :param warning: Warn if this method is called multiple times
    :return:
    """

    if not is_distributed_available():
        raise RuntimeError('Distributed training is not available on this machine')

    if use_horovod:
        global IS_DISTRIBUTED_HOROVOD
        IS_DISTRIBUTED_HOROVOD = True
        if backend is not None or init_method is not None:
            raise RuntimeError('Try to use horovod, but `backend` and `init_method` are not None')

        if is_horovod_available():
            import horovod.torch as hvd

            hvd.init()
            logger.info("Horovod initialized")
        else:
            raise RuntimeError('horovod is not available!')

    else:
        # default values
        backend = backend or "nccl"
        init_method = init_method or "env://"

        if not is_distributed():
            raise RuntimeError(
                f"For distributed training, use `python -m torch.distributed.launch "
                f"--nproc_per_node={device_count()} {get_args()}` ...")

        if distributed.is_initialized():
            if warning:
                logger.warn("`distributed` is already initialized. Skipped.")
        else:
            distributed.init_process_group(backend=backend, init_method=init_method)
        logger.info("Distributed initialized")

    if not is_master():
        def no_print(*values, **kwargs):
            pass

        builtins.print = no_print


def if_is_master(func: Callable
                 ) -> Callable:
    """ Wraps void functions that are active only if it is the master process::

    @if_is_master
    def print_master(message):
        print(message)

    :param func: Any function
    """

    @wraps(func)
    def inner(*args, **kwargs) -> None:
        if is_master():
            return func(*args, **kwargs)

    return inner


import tqdm


def _tqdm(iter, *args, **kwargs):
    if is_master():
        return tqdm.tqdm(iter, *args, **kwargs)
    else:
        return iter


if is_distributed():
    logger.info("tqdm is active only on the master")
    tqdm.tqdm = _tqdm
