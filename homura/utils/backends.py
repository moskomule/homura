""" Helper functions to convert PyTorch Tensors <->  Cupy/Numpy arrays. These functions  are useful to write device-agnostic extensions.
"""

import numpy as np
import torch
from torch.utils.dlpack import from_dlpack, to_dlpack

from .environment import is_cupy_available, is_opteinsum_available

has_cupy = is_cupy_available()
if has_cupy:
    import cupy

has_opt_einsum = is_opteinsum_available()
if has_opt_einsum:
    import opt_einsum


def torch_to_xp(input: torch.Tensor
                ) -> np.ndarray:
    """ Convert a PyTorch tensor to a Cupy/Numpy array.
    """

    if not isinstance(input, torch.Tensor):
        raise RuntimeError(f'torch_to_numpy expects torch.Tensor as input, but got {type(input)}')

    if has_cupy and input.is_cuda:
        return cupy.fromDlpack(to_dlpack(input))
    else:
        return input.numpy()


def xp_to_torch(input: np.ndarray
                ) -> torch.Tensor:
    """ Convert a Cupy/Numpy array to a PyTorch tensor
    """

    if isinstance(input, np.ndarray):
        return torch.from_numpy(input)
    elif has_cupy and isinstance(input, cupy.ndarray):
        return from_dlpack(cupy.ToDlpack(input))
    else:
        raise RuntimeError(f'xp_to_torch expects numpy/cupy.ndarray as input, but got {type(input)}')


def einsum(expr: str,
           *xs):
    if has_opt_einsum:
        return opt_einsum.contract(expr, *xs, backend='torch')
    return torch.einsum(expr, *xs)
