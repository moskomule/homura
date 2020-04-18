import torch
from torch import nn, distributed
from torch.nn import functional as F

from homura.utils import is_faiss_available, is_distributed, is_horovod_available
from .ema import exponential_moving_average_
from .functional import custom_straight_through_estimator, k_nearest_neighbor as knn


class VQModule(nn.Module):
    """ Vector Quantization module used in VQ-VAE [van den Oord et al. 17]

    """

    def __init__(self,
                 emb_dim: int,
                 dict_size: int,
                 momentum: float = 0.99,
                 epsilon: float = 1e-5,
                 knn_backend="faiss" if is_faiss_available() else "torch",
                 metric: str = 'l2'):

        super(VQModule, self).__init__()

        self.emb_dim = emb_dim
        self.dict_size = dict_size
        self.epsilon = epsilon
        self._knn_backend = knn_backend
        self.metric = metric
        self.frozen = False
        # this handles the issue with DataParallel

        assert 0 <= momentum <= 1
        self.gamma = momentum

        # embed: DxC (emb_dim==C)
        embed = F.normalize(torch.randn(dict_size, emb_dim), dim=1, p=2)
        self.register_buffer("track_num", torch.zeros(dict_size, 1))
        self.register_buffer("track_enc", embed.clone())
        self.register_buffer("embed", embed)
        self._first_time = True

        self._distributed_update()

    def forward(self,
                input: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        # returns reconstructed inputs and corresponding reconstructed loss

        distance, ids, vqs = self._forward(*self.flatten(input))
        return custom_straight_through_estimator(vqs, input), F.mse_loss(vqs.detach(), input), ids

    @torch.no_grad()
    def _forward(self,
                 flatten: torch.Tensor,
                 shape: tuple) -> (torch.Tensor, torch.Tensor, torch.Tensor):

        # distance: (BWH)x1, ids: (BWH)x1
        distance, ids = knn(self.embed, flatten, 1, self.metric, backend=self._knn_backend)

        vqs = self.lookup(ids)

        if self.training and not self.frozen:
            self.ema_update(flatten, ids)

        if len(shape) == 4:
            # vqs: (BWH)xC -> BxCxHxW
            b, c, h, w = shape
            vqs = vqs.view(b, w, h, c).transpose(1, -1)
            ids = ids.view(b, w, h).transpose(1, -1)
        else:
            # vqs
            vqs = vqs.squeeze()

        return distance, ids, vqs

    def flatten(self,
                input: torch.Tensor):
        if input.dim() == 2:
            # BxC
            shape = input.size()
            flatten = input
        elif input.dim() == 4:
            # input: BxCxHxW -> flatten: (BWH)xC
            shape = input.size()
            flatten = input.transpose(1, -1).reshape(-1, self.emb_dim)
        else:
            raise NotImplementedError
        return flatten, shape

    @torch.no_grad()
    def ema_update(self,
                   flatten: torch.Tensor,
                   ids: torch.Tensor) -> None:
        # flatten: (BHW)xC, ids: BxHxW -> (BHW)x1
        ids = ids.view(-1, 1)
        # onehot_ids: (BHW)xD
        onehot_ids = ids.new_zeros([ids.size(0), self.dict_size], dtype=torch.float)
        onehot_ids.scatter_(1, ids, 1)
        # (BHW)xD -> 1xD -> Dx1
        if self._first_time:
            self.track_num.copy_(onehot_ids.sum(dim=0).view_as(self.track_num))
            self.track_enc.copy_(onehot_ids.t().matmul(flatten))
            self._first_time = False
        else:
            exponential_moving_average_(self.track_num,
                                        onehot_ids.sum(dim=0).view_as(self.track_num),
                                        self.gamma)
            # Dx(BHW) x (BHW)xC -> DxC
            exponential_moving_average_(self.track_enc,
                                        onehot_ids.t().matmul(flatten),
                                        self.gamma)

        # following sonnet's implementation
        factor = 1 + (self.epsilon * self.dict_size) / self.track_num.sum()
        self.embed = self.track_enc * factor / (self.track_num + self.epsilon)
        self._distributed_update()

    def _distributed_update(self):
        if not is_distributed():
            return

        if is_horovod_available():
            import horovod.torch as hvd

            hvd.allreduce(self.track_num)
            hvd.allreduce(self.track_enc)
            hvd.allreduce(self.embed)
        else:
            distributed.all_reduce(self.track_num, op=distributed.ReduceOp.SUM)
            distributed.all_reduce(self.track_enc, op=distributed.ReduceOp.SUM)
            distributed.all_reduce(self.embed, op=distributed.ReduceOp.SUM)
            ws = distributed.get_world_size()
            self.track_num /= ws
            self.track_enc /= ws
            self.embed /= ws

    def lookup(self, ids: torch.Tensor):
        return F.embedding(ids, self.embed)

    def __repr__(self):
        return f"VQModule(emb_dim={self.emb_dim}, dict_size={self.dict_size})"
