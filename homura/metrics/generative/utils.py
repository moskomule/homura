import math

import torch
from torch.nn import functional as F

from homura.modules.functional.loss import _reduction

"""
 gan: GAN's expectation
 js: Jansen Shanon
 kl: Kullback Leibler
 dv: Donsker Varahdan
"""
_positive_dic = {"gan": lambda x: -F.softplus(-x),
                 "js": lambda x: math.log(2) - F.softplus(-x),
                 "kl": lambda x: x + 1,
                 "dv": lambda x: x}

_negative_dic = {"gan": lambda x: F.softplus(-x) + x,
                 "js": lambda x: F.softplus(-x) + x - math.log(2),
                 "kl": lambda x: x.exp(),
                 "dv": lambda x: x.logsumexp(0) - math.log(x.size(0))}


def _positive_expectation(input: torch.Tensor,
                          measure: str,
                          reduction: str = "mean"):
    return _reduction(_positive_dic[measure](input), reduction)


def _negative_expectation(input: torch.Tensor,
                          measure: str,
                          reduction: str = "mean"):
    return _reduction(_negative_dic[measure](input), reduction)
