from __future__ import annotations

import torch

from homura.utils import is_faiss_available

if is_faiss_available():
    import faiss


def _tensor_to_ptr(input: torch.Tensor):
    assert input.is_contiguous()
    assert input.dtype in {torch.float32, torch.int64}
    if input.dtype is torch.float32:
        return faiss.cast_integer_to_float_ptr(input.storage().data_ptr() + input.storage_offset() * 4)
    else:
        return faiss.cast_integer_to_long_ptr(input.storage().data_ptr() + input.storage_offset() * 8)


@torch.no_grad()
def torch_knn(keys: torch.Tensor,
              queries: torch.Tensor,
              num_neighbors: int,
              distance: str
              ) -> tuple[torch.Tensor, torch.Tensor]:
    """ k nearest neighbor using torch. Users are recommended to use `k_nearest_neighbor` instead.
    """

    if distance == 'inner_product':
        # for JIT
        keys = torch.nn.functional.normalize(keys, p=2.0, dim=1)
        queries = torch.nn.functional.normalize(queries, p=2.0, dim=1)
        scores = keys.mm(queries.t())
    else:
        scores = keys.mm(queries.t())
        scores *= 2
        scores -= (keys.pow(2)).sum(1, keepdim=True)
        scores -= (queries.pow(2)).sum(1).unsqueeze_(0)
    scores, indices = scores.topk(k=num_neighbors, dim=0, largest=True)
    scores = scores.t()
    indices = indices.t()

    return scores, indices


@torch.no_grad()
def faiss_knn(keys: torch.Tensor,
              queries: torch.Tensor,
              num_neighbors: int,
              distance: str
              ) -> tuple[torch.Tensor, torch.Tensor]:
    """ k nearest neighbor using faiss. Users are recommended to use `k_nearest_neighbor` instead.

    :param keys: tensor of (num_keys, dim)
    :param queries: tensor of (num_queries, dim)
    :param num_neighbors: `k`
    :param distance: user can use str or faiss.METRIC_*.
    :return: scores, indices in tensor
    """

    if not is_faiss_available():
        raise RuntimeError("_faiss_knn requires faiss-gpu")

    metric_map = {"inner_product": faiss.METRIC_INNER_PRODUCT,
                  "l2": faiss.METRIC_L2,
                  "l1": faiss.METRIC_L1,
                  "linf": faiss.METRIC_Linf,
                  "jansen_shannon": faiss.METRIC_JensenShannon}

    k_ptr = _tensor_to_ptr(keys)
    q_ptr = _tensor_to_ptr(queries)

    scores = keys.new_empty((queries.size(0), num_neighbors), dtype=torch.float32)
    indices = keys.new_empty((queries.size(0), num_neighbors), dtype=torch.int64)

    s_ptr = _tensor_to_ptr(scores)
    i_ptr = _tensor_to_ptr(indices)

    args = faiss.GpuDistanceParams()
    args.metric = metric_map[distance] if isinstance(distance, str) else distance
    args.k = num_neighbors
    args.dims = queries.size(1)
    args.vectors = k_ptr
    args.vectorsRowMajor = True
    args.numVectors = keys.size(0)
    args.queries = q_ptr
    args.queriesRowMajor = True
    args.numQueries = queries.size(0)
    args.outDistances = s_ptr
    args.outIndices = i_ptr
    faiss.bfKnn(FAISS_RES, args)
    return scores, indices


def k_nearest_neighbor(keys: torch.Tensor,
                       queries: torch.Tensor,
                       num_neighbors: int,
                       distance: str, *,
                       backend: str = "torch") -> tuple[torch.Tensor, torch.Tensor]:
    """ k-Nearest Neighbor search. Faiss backend requires GPU. torch backend is JITtable

    :param keys: tensor of (num_keys, dim)
    :param queries: tensor of (num_queries, dim)
    :param num_neighbors: `k`
    :param distance: name of distance (`inner_product` or `l2`). Faiss backend additionally supports `l1`, `linf`, `jansen_shannon`.
    :param backend: backend (`faiss` or `torch`)
    :return: scores, indices
    """

    assert backend in {"faiss", "torch", "torch_jit"}
    assert keys.size(1) == queries.size(1)
    assert keys.ndim == 2 and queries.ndim == 2

    f = faiss_knn if backend == "faiss" and is_faiss_available() else torch_knn
    return f(keys, queries, num_neighbors, distance)


if is_faiss_available():
    import faiss

    FAISS_RES = faiss.StandardGpuResources()
    FAISS_RES.setDefaultNullStreamAllDevices()
    FAISS_RES.setTempMemory(1200 * 1024 * 1024)
