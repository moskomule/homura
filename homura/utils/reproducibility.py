import contextlib
import random
from typing import Optional

import numpy
import torch

from homura import get_global_rank
from homura.liblog import get_logger

logger = get_logger(__name__)


@contextlib.contextmanager
def set_seed(seed: Optional[int] = None):
    """ Fix seed of random generator in the given context. ::

        >>> with set_seed(0):
        >>>     do_some_random_thing()

    """

    s_py, s_np, s_torch = random.getstate(), numpy.random.get_state(), torch.get_rng_state()
    if torch.cuda.is_available():
        s_cuda = torch.cuda.get_rng_state_all()
    if seed is not None:
        # to avoid using the same seed on different processes
        seed += get_global_rank()
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        logger.info(f"Set seed to {seed}")
    yield
    # recover random states
    random.seed(s_py)
    numpy.random.set_state(s_np)
    torch.set_rng_state(s_torch)
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s_cuda)


@contextlib.contextmanager
def set_deterministic(seed: Optional[int] = None):
    """ Set seed of `torch`, `random` and `numpy` to `seed` for making it deterministic. Because of CUDA's limitation, this
    does not make everything deterministic, however.
    """

    with set_seed(seed):
        if seed is not None:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.info("Set deterministic. But some GPU computations might be still non-deterministic. "
                        "Also, this may affect the performance.")
        yield
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    logger.info("Back to non-deterministic.")
