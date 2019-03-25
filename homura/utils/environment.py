import importlib.util

from homura.liblog import get_logger

__all__ = ["is_accimage_available", "is_apex_available", "is_tensorboardX_available",
           "enable_accimage"]

logger = get_logger("homura.env")
is_accimage_available = importlib.util.find_spec("accimage") is not None
is_apex_available = importlib.util.find_spec("apex") is not None
is_tensorboardX_available = importlib.util.find_spec("tensorboardX") is not None


def enable_accimage():
    if is_accimage_available:
        import torchvision

        torchvision.set_image_backend("accimage")
    else:
        logger.warning("accimage is not available")
