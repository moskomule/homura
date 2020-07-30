from typing import Tuple

import torch

from homura.utils import is_faiss_available


def _tensor_to_ptr(input: torch.Tensor):
    import faiss

    assert input.is_contiguous()
    assert input.dtype in [torch.float32, torch.int64]
    if input.dtype is torch.float32:
        return faiss.cast_integer_to_float_ptr(input.storage().data_ptr() + input.storage_offset() * 4)
    else:
        return faiss.cast_integer_to_long_ptr(input.storage().data_ptr() + input.storage_offset() * 8)


def _torch_knn(keys: torch.Tensor,
               queries: torch.Tensor,
               num_neighbors: int,
               distance: str) -> Tuple[torch.Tensor, torch.Tensor]:
    assert distance in ['dot_product', 'l2']
    assert keys.size(1) == queries.size(1)

    with torch.no_grad():
        if distance == "dot_product":
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


def _faiss_knn(keys: torch.Tensor,
               queries: torch.Tensor,
               num_neighbors: int,
               distance: str) -> Tuple[torch.Tensor, torch.Tensor]:
    # https://github.com/facebookresearch/XLM/blob/master/src/model/memory/utils.py
    if not is_faiss_available():
        raise RuntimeError("faiss_knn requires faiss-gpu")
    import faiss

    assert distance in ['dot_product', 'l2']
    assert keys.size(1) == queries.size(1)

    metric = faiss.METRIC_INNER_PRODUCT if distance == 'dot_product' else faiss.METRIC_L2

    k_ptr = _tensor_to_ptr(keys)
    q_ptr = _tensor_to_ptr(queries)

    scores = keys.new_zeros((queries.size(0), num_neighbors), dtype=torch.float32)
    indices = keys.new_zeros((queries.size(0), num_neighbors), dtype=torch.int64)

    s_ptr = _tensor_to_ptr(scores)
    i_ptr = _tensor_to_ptr(indices)

    faiss.bfKnn(FAISS_RES, metric,
                k_ptr, True, keys.size(0),
                q_ptr, True, queries.size(0),
                queries.size(1), num_neighbors, s_ptr, i_ptr)
    return scores, indices


def k_nearest_neighbor(keys: torch.Tensor,
                       queries: torch.Tensor,
                       num_neighbors: int,
                       distance: str, *,
                       backend: str = "torch") -> Tuple[torch.Tensor, torch.Tensor]:
    """ k-Nearest Neighbor search

    :param keys: tensor of (num_keys, dim)
    :param queries: tensor of (num_queries, dim)
    :param num_neighbors: `k`
    :param distance: registry_name of distance (`dot_product` or `l2`)
    :param backend: backend (`faiss` or `torch`)
    :return: scores, indices
    """
    assert backend in ["faiss", "torch"]
    f = _faiss_knn if backend == "faiss" and is_faiss_available() else _torch_knn
    return f(keys, queries, num_neighbors, distance)


if is_faiss_available():
    import faiss

    FAISS_RES = faiss.StandardGpuResources()
    FAISS_RES.setDefaultNullStreamAllDevices()
    FAISS_RES.setTempMemory(1200 * 1024 * 1024)
