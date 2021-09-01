from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.autograd import grad


def param_to_vector(parameters: Iterable[torch.Tensor]) -> torch.Tensor:
    devices = [p.data.device for p in parameters]
    if len(devices) > 1 and any([devices[0] != d for d in devices[1:]]):
        raise RuntimeError("Inconsistent devices in parameters!")

    return torch.cat([param.reshape(-1) for param in parameters])


def vjp(f: torch.Tensor,
        p: nn.Parameter | list[nn.Parameter],
        v: torch.Tensor,
        *,
        only_retain_graph: bool = False
        ) -> torch.Tensor:
    """ vector Jacobian product
    """

    return param_to_vector(grad(f, p, v, create_graph=not only_retain_graph, retain_graph=True))  # dim p


def jvp(f: torch.Tensor,
        p: nn.Parameter | list[nn.Parameter],
        v: torch.Tensor
        ) -> torch.Tensor:
    """ Jacobian vector product
    """

    dummy = torch.ones_like(f, requires_grad=True)
    g = vjp(f, p, dummy)
    # note that we don't need higher order gradient w.r.t. dummy
    return vjp(g, dummy, v, only_retain_graph=True)  # dim y


def hvp(loss: torch.Tensor,
        f,
        p: nn.Parameter or list[nn.Parameter],
        v: torch.Tensor
        ) -> torch.Tensor:
    """ Hessian vector product
    """

    df_dp = param_to_vector(grad(loss, p, create_graph=True))
    return vjp(df_dp, p, v)  # dim p


def ggnvp(loss: torch.Tensor,
          f: torch.Tensor,
          p: nn.Parameter | list[nn.Parameter],
          v: torch.Tensor
          ) -> torch.Tensor:
    """ Generalized Gaussian Newton vector product.  In case of loss=F.cross_entropy(output, target),
    GGN matrix is equivalent to the Fisher matrix.
    """

    jv = jvp(f, p, v)  # dim y
    hjv = hvp(loss, None, f, jv)  # dim y
    jhjv = vjp(f.view(-1), p, hjv)  # dim p
    return jhjv
