import torch

__all__ = ["cross_entropy_with_softlabels"]


def _reduction(input: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "mean":
        return input.mean()
    elif reduction == "sum":
        return input.sum()
    elif reduction == "none" or reduction is None:
        return input
    else:
        raise NotImplementedError(f"Wrong reduction: {reduction}")


def cross_entropy_with_softlabels(input: torch.Tensor,
                                  target: torch.Tensor,
                                  dim: int = 1,
                                  reduction: str = "mean") -> torch.Tensor:
    """ Cross entropy with soft labels. Unlike `torch.nn.functional.cross_entropy`, `target` is expected to be
    one-hot or soft label.

    :param input: Tensor of `BxCx(optional dimensions)`
    :param target: Tensor of `BxCx(optional dimensions)`
    :param reduction:
    :return:
    """
    if input.size() != target.size():
        raise RuntimeError(f"Input size ({input.size()}) and target size ({target.size()}) should be same!")
    return _reduction(-(input.log_softmax(dim=dim) * target).sum(dim=dim), reduction)
