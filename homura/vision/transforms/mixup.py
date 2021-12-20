from __future__ import annotations

import torch


def partial_mixup(input: torch.Tensor,
                  gamma: float,
                  indices: torch.Tensor
                  ) -> torch.Tensor:
    """ mixup: Beyond Empirical Risk Minimization

    :param input:
    :param gamma:
    :param indices:
    :return:
    """

    if input.size(0) != indices.size(0):
        raise RuntimeError("Size mismatch!")
    perm_input = input[indices]
    return input.mul(gamma).add(perm_input, alpha=1 - gamma)


def mixup(input: torch.Tensor,
          target: torch.Tensor,
          gamma: float,
          indices: torch.Tensor = None
          ) -> tuple[torch.Tensor, torch.Tensor]:
    """ mixup: Beyond Empirical Risk Minimization

    :param input:
    :param target:
    :param gamma:
    :param indices:
    :return:
    """

    if indices is None:
        indices = torch.randperm(input.size(0), device=input.device, dtype=torch.long)
    return partial_mixup(input, gamma, indices), partial_mixup(target, gamma, indices)
