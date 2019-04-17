import contextlib
import random

import numpy
import torch

from homura.liblog import get_logger

logger = get_logger(__name__)


@contextlib.contextmanager
def set_seed(seed: int = 0):
    """ Fix seed of random generator in the given context. ::

        >>> with set_seed(0):
        >>>     do_some_random_thing()

    """

    random.seed(seed)
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    yield
    random.seed()
    new_seed = random.randrange(2 ** 32 - 1)
    torch.manual_seed(new_seed)
    new_seed = random.randrange(2 ** 32 - 1)
    numpy.random.seed(new_seed)


@contextlib.contextmanager
def set_deterministic(seed: int = 0):
    """ Set seed of `torch`, `random` and `numpy` to 0 for making it deterministic. Because of CUDA's limitation, this
    does not make everything deterministic, however.
    """

    with set_seed(seed):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info("Set deterministic. But some GPU computations might be still non-deterministic. "
                    "Also, this may affect the performance.")
        yield
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    logger.info("Back to non-deterministic.")
