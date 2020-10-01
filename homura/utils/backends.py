""" Helper functions to convert PyTorch Tensors <->  Cupy/Numpy arrays. These functions  are useful to write device-agnostic extensions.
"""

import numpy as np
import torch
from torch.utils.dlpack import from_dlpack, to_dlpack

from .environment import is_cupy_available

IS_CUPY_AVAILABLE = is_cupy_available()
if IS_CUPY_AVAILABLE:
    import cupy


def torch_to_xp(input: torch.Tensor
                ) -> np.ndarray:
    """ Convert a PyTorch tensor to a Cupy/Numpy array.
    """

    if not torch.is_tensor(input):
        raise RuntimeError(f'torch_to_numpy expects torch.Tensor as input, but got {type(input)}')

    if IS_CUPY_AVAILABLE and input.is_cuda:
        return cupy.fromDlpack(to_dlpack(input))
    else:
        return input.numpy()


def xp_to_torch(input: np.ndarray
                ) -> torch.Tensor:
    """ Convert a Cupy/Numpy array to a PyTorch tensor
    """

    if isinstance(input, np.ndarray):
        return torch.from_numpy(input)
    elif IS_CUPY_AVAILABLE and isinstance(input, cupy.ndarray):
        return from_dlpack(cupy.ToDlpack(input))
    else:
        raise RuntimeError(f'xp_to_torch expects numpy/cupy.ndarray as input, but got {type(input)}')
