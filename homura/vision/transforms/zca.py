import torch
from torchvision.transforms import LinearTransformation


def zca_statistics(input: torch.Tensor,
                   eps: float = 1e-3):
    """ ZCA-whitening

    :param input:
    :param eps:
    :return:
    """
    if input.dim() != 4:
        raise RuntimeError(f"Dimension of `input` is expected to be 4 but got {input.dim()}")
    b, c, h, w = input.size()
    input = input.reshape(b, -1)
    mean = input.mean(dim=0)
    input -= mean
    cov = input.t().matmul(input) / b
    u, s, v = cov.svd(compute_uv=True)
    matrix = (u @ (s + eps).sqrt_().reciprocal_().diag()) @ u.t()
    return matrix, mean


class ZCATransformation(LinearTransformation):
    """

    >>> transform = ZCATransformation.create(torch.randn(4, 3, 32, 32))
    >>> transform(torch.randn(2, 3, 32, 32))
    """

    @staticmethod
    def create(input: torch.Tensor,
               eps: float = 1e-3):
        matrix, mean = zca_statistics(input, eps)
        return ZCATransformation(matrix, mean)
