# get information on the environment

import importlib.util
import os as python_os
import subprocess
import sys as python_sys

from torch import distributed
from torch.cuda import device_count

from homura.liblog import get_logger

logger = get_logger("homura.env")
args = " ".join(python_sys.argv)
_DISTRIBUTED_FLAG = False


# Utility functions that useful libraries are available or not
def is_accimage_available() -> bool:
    return importlib.util.find_spec("accimage") is not None


def is_horovod_available() -> bool:
    disable_horovod = int(get_environ("HOMURA_DISABLE_HOROVOD", 0))
    return (importlib.util.find_spec("horovod") is not None) and (disable_horovod == 0)


def is_faiss_available() -> bool:
    try:
        import faiss

        return hasattr(faiss, 'StandardGpuResources')
    except ImportError:
        return False


def is_distributed_available() -> bool:
    return (hasattr(distributed, "is_available") and getattr(distributed, "is_available")) or is_horovod_available()


def is_distributed() -> bool:
    return get_world_size() > 1


def get_git_hash() -> str:
    def _decode_bytes(b: bytes) -> str:
        return b.decode("ascii")[:-1]

    try:
        is_git_repo = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"],
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout
    except FileNotFoundError:
        return ""

    if _decode_bytes(is_git_repo) == "true":
        git_hash = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                  stdout=subprocess.PIPE).stdout
        return _decode_bytes(git_hash)
    else:
        logger.info("No git info available in this directory")
        return ""


def get_args() -> list:
    return python_sys.argv


def get_environ(name: str,
                default) -> str:
    return python_os.environ.get(name, default)


def get_local_rank() -> int:
    # returns -1 if not distributed, else returns local rank
    # it works before dist.init_process_group
    if not is_distributed():
        return -1
    else:
        if is_horovod_available():
            import horovod.torch as hvd

            return hvd.local_rank()
        for arg in python_sys.argv:
            if "--local_rank" in arg:
                return int(arg.split("=")[1])


def get_global_rank() -> int:
    # returns -1 if not distributed, else returns global rank
    # it works before dist.init_process_group
    if _DISTRIBUTED_FLAG and is_horovod_available():
        import horovod.torch as hvd

        return hvd.rank()
    return int(get_environ('RANK', -1))


def is_master() -> bool:
    return get_global_rank() <= 0


def get_num_nodes() -> int:
    # assume all nodes have the same number of gpus
    if not is_distributed():
        return 1
    else:
        return get_world_size() // device_count()


def get_world_size() -> int:
    if _DISTRIBUTED_FLAG and is_horovod_available():
        import horovod.torch as hvd

        return hvd.size()
    return int(python_os.environ.get("WORLD_SIZE", 1))


def init_distributed(use_horovod: bool = False, *,
                     backend="nccl",
                     init_method="env://",
                     warning=True):
    # A simple initializer of distributed

    if not is_distributed_available():
        raise RuntimeError('Distributed training is not available on this machine')

    global _DISTRIBUTED_FLAG
    _DISTRIBUTED_FLAG = True
    if use_horovod:
        if is_horovod_available():
            import horovod.torch as hvd

            hvd.init()
            logger.debug("init horovod")
        else:
            raise RuntimeError('horovod is not available!')

    else:

        if not is_distributed():
            raise RuntimeError(
                f"For distributed training, use `python -m torch.distributed.launch "
                f"--nproc_per_node={device_count()} {args}` ...")

        if distributed.is_initialized():
            if warning:
                logger.warn("`distributed` is already initialized. Skipped.")
        else:
            distributed.init_process_group(backend=backend, init_method=init_method)
        logger.debug("init distributed")


def enable_accimage() -> None:
    if is_accimage_available():
        import torchvision

        torchvision.set_image_backend("accimage")
    else:
        logger.warning("accimage is not available")
