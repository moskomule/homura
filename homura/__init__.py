import importlib.util

from . import utils, modules, vision, lr_scheduler, optim, liblog
from .liblog import get_logger

__all__ = ["utils", "modules", "vision", "lr_scheduler", "optim", "liblog", "debug"]

logger = get_logger(__name__)
is_apex_available = False
if importlib.util.find_spec("apex") is not None:
    is_apex_available = True
    logger.debug("apex is available")
else:
    logger.debug("apex is unavailable")
