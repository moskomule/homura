import warnings
from functools import partial

import torch
from torch import nn

from .functional import cross_entropy_with_smoothing, cross_entropy_with_softlabels


class _LossFunction(nn.Module):
    def forward(self,
                input: torch.Tensor,
                target: torch.Tensor
                ) -> torch.Tensor:
        return self.impl(input, target)


class SoftLabelCrossEntropy(_LossFunction):
    def __init__(self,
                 dim: int = 1,
                 reduction: str = "mean"):
        super().__init__()
        if hasattr(nn.CrossEntropyLoss, "label_smoothing"):
            warnings.warn("Use PyTorch's nn.CrossEntropyLoss", DeprecationWarning)
        self.impl = partial(cross_entropy_with_softlabels, dim=dim, reduction=reduction)


class SmoothedCrossEntropy(_LossFunction):
    def __init__(self,
                 smoothing: float = 0.1,
                 dim: int = 1,
                 reduction: str = "mean"):
        super().__init__()
        if hasattr(nn.CrossEntropyLoss, "label_smoothing"):
            warnings.warn("Use PyTorch's nn.CrossEntropyLoss", DeprecationWarning)
        self.impl = partial(cross_entropy_with_smoothing, smoothing=smoothing, dim=dim, reduction=reduction)
