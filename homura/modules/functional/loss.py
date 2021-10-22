import warnings

import torch


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
    """

    :param input:
    :param target:
    :param dim:
    :param reduction:
    :return:
    """
    if hasattr(torch.nn.CrossEntropyLoss, "label_smoothing"):
        warnings.warn("Use PyTorch's F.cross_entropy", DeprecationWarning)
    if input.size() != target.size():
        raise RuntimeError(f"Input size ({input.size()}) and target size ({target.size()}) should be same!")
    return _reduction(-(input.log_softmax(dim=dim) * target).sum(dim=dim), reduction)


def cross_entropy_with_smoothing(input: torch.Tensor,
                                 target: torch.Tensor,
                                 smoothing: float,
                                 dim: int = 1,
                                 reduction: str = "mean"
                                 ) -> torch.Tensor:
    """

    :param input:
    :param target:
    :param smoothing:
    :param dim:
    :param reduction:
    :return:
    """

    if hasattr(torch.nn.CrossEntropyLoss, "label_smoothing"):
        warnings.warn("Use PyTorch's F.cross_entropy", DeprecationWarning)
    log_prob = input.log_softmax(dim=dim)
    nll_loss = -log_prob.gather(dim=dim, index=target.unsqueeze(dim=dim))
    nll_loss = nll_loss.squeeze(dim=dim)
    smooth_loss = -log_prob.mean(dim=dim)
    loss = (1 - smoothing) * nll_loss + smoothing * smooth_loss
    return _reduction(loss, reduction)
