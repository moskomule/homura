import contextlib
import random

import numpy
import torch

from homura.liblog import get_logger
from homura.utils.distributed import get_global_rank

logger = get_logger(__name__)


@contextlib.contextmanager
def set_seed(seed: int = None,
             by_rank: bool = False):
    """ Fix seed of random generator in the given context. ::

        >>> with set_seed(0):
        >>>     do_some_random_thing()

    """

    s_py, s_np, s_torch = random.getstate(), numpy.random.get_state(), torch.get_rng_state()
    if torch.cuda.is_available():
        s_cuda = torch.cuda.get_rng_state_all()
    if seed is not None:
        # to avoid using the same seed on different processes
        if by_rank:
            seed += get_global_rank()
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        # these functions are safe even if cuda is not available
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        logger.info(f"Set seed to {seed}")
    yield
    # recover random states
    random.setstate(s_py)
    numpy.random.set_state(s_np)
    torch.set_rng_state(s_torch)
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s_cuda)


@contextlib.contextmanager
def set_deterministic(seed: int = None,
                      by_rank: bool = False):
    """ Set seed of `torch`, `random` and `numpy` to `seed` for making it deterministic. Because of CUDA's limitation,
    this may not make everything deterministic, however.
    """

    with set_seed(seed, by_rank):
        if seed is not None:
            if hasattr(torch, "set_deterministic"):
                torch.set_deterministic(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.info("Set deterministic. But some GPU computations might be still non-deterministic. "
                        "Also, this may affect the performance.")
        yield
    if hasattr(torch, "set_deterministic"):
        torch.set_deterministic(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    logger.info("Back to non-deterministic.")
