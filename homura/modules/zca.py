import torch
from torch import nn

from homura.vision.transforms.zca import zca_statistics


class ZCA(nn.Module):
    """ ZCA requires large matrix multiplication, so one solution is do it on GPUs.

    >>> zca = ZCA.create(torch.randn(4, 3, 24, 24))
    >>> zca(torch.rand(4, 3, 24, 24))

    """

    def __init__(self, matrix, mean):
        super(ZCA, self).__init__()
        self.register_buffer("matrix", matrix)
        self.register_buffer("mean", mean)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() != 4:
            raise RuntimeError(f"Dimension of `input` is expected to be 4 but got {input.dim()}")

        flat_tensor = input.view(input.size(0), -1) - self.mean
        transformed_tensor = flat_tensor @ self.matrix
        return transformed_tensor.view_as(input)

    @staticmethod
    def create(input: torch.Tensor,
               eps: float = 1e-3):
        matrix, mean = zca_statistics(input, eps)
        return ZCA(matrix, mean.view(1, -1))
