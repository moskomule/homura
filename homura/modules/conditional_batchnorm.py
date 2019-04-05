import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.batchnorm import _BatchNorm


class CategoricalConditionalBatchNorm(_BatchNorm):
    """
    (Categorical) conditional batch normalization, proposed in
    V. Dumoulin, J. Shlens, and M. Kudlur, "A learned representation for artistic style",
    https://arxiv.org/abs/1610.07629 (In this paper, conditional *instance* normalization is used)

    :param num_features:
    :param num_classes:
    :param eps:
    :param momentum:
    :param affine:
    :param track_running_stats:
    """

    def __init__(self,
                 num_features,
                 num_classes,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True):
        super(CategoricalConditionalBatchNorm, self).__init__(num_features, eps, momentum, affine,
                                                              track_running_stats)
        self._gamma_emb = nn.Embedding(num_classes, embedding_dim=num_features)
        self._beta_emb = nn.Embedding(num_classes, embedding_dim=num_features)

    def forward(self,
                inputs: torch.Tensor,
                categories: torch.Tensor) -> torch.Tensor:
        ret = F.batch_norm(inputs, self.running_mean, self.running_var, self.weight, self.bias,
                           self.training or not self.track_running_stats, self.momentum, self.eps)
        gamma = self._gamma_emb(categories)
        beta = self._beta_emb(categories)
        gamma = gamma.view(*gamma.shape, 1, 1)
        beta = beta.view(*beta.shape, 1, 1)
        return gamma * ret + beta
