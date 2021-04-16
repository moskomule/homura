import torch

from homura.modules import k_nearest_neighbor
from homura.modules.functional.knn import torch_knn


def test_knn():
    k = torch.tensor([[-0.8371, -0.1735, -1.1959],
                      [-0.7557, -0.6231, 1.4350],
                      [-0.7415, -0.9763, 0.8254],
                      [0.0977, -0.6569, 0.4799],
                      [0.9849, 0.5081, -1.8797]])
    q = torch.tensor([[-0.5039, -0.2573, -0.1468],
                      [0.2302, -0.6094, 1.6183],
                      [-0.3824, -0.4617, 0.1872],
                      [-1.1542, -0.4422, 0.0420]])
    expected = [3, 1, 3, 2]
    assert k_nearest_neighbor(k, q, 1, "l2")[1].view(-1).tolist() == expected


def test_torch_knn_jittable():
    torch.jit.script(torch_knn)
