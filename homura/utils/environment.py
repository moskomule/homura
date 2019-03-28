import importlib.util
import os as python_os
import sys as python_sys

from homura.liblog import get_logger

__all__ = ["is_accimage_available", "is_apex_available", "is_tensorboardX_available", "is_distributed",
           "enable_accimage",
           "get_global_rank", "get_local_rank", "get_world_size"]

logger = get_logger("homura.env")
is_accimage_available = importlib.util.find_spec("accimage") is not None
is_apex_available = importlib.util.find_spec("apex") is not None
is_tensorboardX_available = importlib.util.find_spec("tensorboardX") is not None

args = " ".join(python_sys.argv)
is_distributed = "--local_rank" in args


def get_local_rank():
    # returns -1 if not distributed, else returns local rank
    # it works before dist.init_process_group
    if not is_distributed:
        return -1
    else:
        for arg in python_sys.argv:
            if "--local_rank" in arg:
                return int(arg.split("=")[1])


def get_global_rank():
    # returns -1 if not distributed, else returns global rank
    # it works before dist.init_process_group
    if not is_distributed:
        return -1
    else:
        return int(python_os.environ["RANK"])


def get_world_size():
    if not is_distributed:
        return 1
    else:
        return int(python_os.environ["WORLD_SIZE"])


def enable_accimage():
    if is_accimage_available:
        import torchvision

        torchvision.set_image_backend("accimage")
    else:
        logger.warning("accimage is not available")
