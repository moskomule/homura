import importlib.util
import os as python_os
import subprocess
import sys as python_sys

from torch import distributed
from torch.cuda import device_count

from homura.liblog import get_logger

logger = get_logger("homura.env")
args = " ".join(python_sys.argv)


def is_accimage_available() -> bool:
    return importlib.util.find_spec("accimage") is not None


def is_apex_available() -> bool:
    return importlib.util.find_spec("apex") is not None


def is_faiss_available() -> bool:
    try:
        import faiss

        return hasattr(faiss, 'StandardGpuResources')
    except ImportError:
        return False


def get_world_size() -> int:
    return int(python_os.environ.get("WORLD_SIZE", 1))


def is_distributed() -> bool:
    return get_world_size() > 1


def is_distributed_avaiable() -> bool:
    return hasattr(distributed, "is_available") and getattr(distributed, "is_available")


def _decode_bytes(b: bytes) -> str:
    return b.decode("ascii")[:-1]


def get_git_hash() -> str:
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


def get_local_rank() -> int:
    # returns -1 if not distributed, else returns local rank
    # it works before dist.init_process_group
    if not is_distributed():
        return -1
    else:
        for arg in python_sys.argv:
            if "--local_rank" in arg:
                return int(arg.split("=")[1])


def get_global_rank() -> int:
    # returns -1 if not distributed, else returns global rank
    # it works before dist.init_process_group
    if not is_distributed():
        return -1
    else:
        return int(python_os.environ["RANK"])


def is_master() -> bool:
    return get_global_rank() <= 0


def get_num_nodes() -> int:
    # assume all nodes have the same number of gpus
    if not is_distributed():
        return 1
    else:
        return get_world_size() // device_count()


def init_distributed(backend="nccl",
                     init_method="env://", *,
                     warning=True):
    # A simple initializer of distributed

    from torch import distributed

    if not distributed.is_available():
        raise RuntimeError("`distributed` is not available.")

    if not is_distributed():
        raise RuntimeError(
            f"For distributed training, use `python -m torch.distributed.launch "
            f"--nproc_per_node={device_count()} {args}` ...")

    if distributed.is_initialized():
        if warning:
            logger.warn("`distributed` is already initialized, so skipped.")
    else:
        distributed.init_process_group(backend=backend, init_method=init_method)


def enable_accimage() -> None:
    if is_accimage_available():
        import torchvision

        torchvision.set_image_backend("accimage")
    else:
        logger.warning("accimage is not available")
