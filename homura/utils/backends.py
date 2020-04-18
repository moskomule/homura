import warnings

import numpy as np
import torch
from torch.utils.dlpack import to_dlpack, from_dlpack

try:
    import cupy

    IS_CUPY_AVAILABLE = True
except ImportError as e:
    IS_CUPY_AVAILABLE = False
    warnings.warn("cupy is not available")


def torch_to_xp(input: torch.Tensor
                ) -> np.ndarray:
    # torch Tensor to numpy/cupy ndarray
    if not torch.is_tensor(input):
        raise RuntimeError(f'torch_to_numpy expects torch.Tensor as input, but got {type(input)}')

    if IS_CUPY_AVAILABLE and input.is_cuda:
        return cupy.fromDlpack(to_dlpack(input))
    else:
        return input.numpy()


def xp_to_torch(input: np.ndarray
                ) -> torch.Tensor:
    # numpy/cupy ndarray to torchTensor
    if isinstance(input, np.ndarray):
        return torch.from_numpy(input)
    elif IS_CUPY_AVAILABLE and isinstance(input, cupy.ndarray):
        return from_dlpack(cupy.ToDlpack(input))
    else:
        raise RuntimeError(f'xp_to_torch expects numpy/cupy.ndarray as input, but got {type(input)}')
