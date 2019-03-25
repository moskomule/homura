import importlib.util
import sys as python_sys

from homura.liblog import get_logger

__all__ = ["is_accimage_available", "is_apex_available", "is_tensorboardX_available", "is_distributed",
           "enable_accimage"]

logger = get_logger("homura.env")
is_accimage_available = importlib.util.find_spec("accimage") is not None
is_apex_available = importlib.util.find_spec("apex") is not None
is_tensorboardX_available = importlib.util.find_spec("tensorboardX") is not None

args = " ".join(python_sys.argv)
is_distributed = "--local_rank" in args


def get_local_rank():
    if not is_distributed:
        return -1
    else:
        for arg in python_sys.argv:
            if "--local_rank" in arg:
                return int(arg.split("=")[1])


def enable_accimage():
    if is_accimage_available:
        import torchvision

        torchvision.set_image_backend("accimage")
    else:
        logger.warning("accimage is not available")
