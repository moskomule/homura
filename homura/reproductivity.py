import random

import numpy
import torch

from homura.liblog import get_logger

logger = get_logger(__name__)


def set_deterministic(seed: int = 0):
    """ Set seed of `torch`, `random` and `numpy` to 0 for making it deterministic. Because of CUDA's limitation, this
    does not make everything deterministic, however.
    """
    random.seed(0)
    torch.manual_seed(0)
    numpy.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Set to be deterministic. But some GPU computations is still nondeterministic. Also, this may affect "
                "the performance.")


def unset_deterministic():
    random.seed()
    new_seed = random.randrange(2 ** 32 - 1)
    torch.manual_seed(new_seed)
    new_seed = random.randrange(2 ** 32 - 1)
    numpy.random.seed(new_seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    logger.info("Set to be non-deterministic.")
