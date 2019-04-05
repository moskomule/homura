import torch
from torch.utils.data import TensorDataset, DataLoader

from homura.vision.data.prefetcher import DataPrefetcher


def test_prefetcher():
    data = torch.randn(128, 3, 32, 32)
    label = torch.arange(128)
    dataset = TensorDataset(data, label)
    loader = DataLoader(dataset, batch_size=32)
    prefetcher = DataPrefetcher(loader)
    for _ in range(2):
        for input, target in prefetcher:
            pass
