import torch
from torch.utils.data import DataLoader

from homura.utils.containers import TensorTuple


class DataPrefetcher(object):
    """ prefetch data
    """

    def __init__(self,
                 loader: DataLoader):
        self._cuda_available = torch.cuda.is_available()
        self._length = len(loader)
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream() if self._cuda_available else None
        self.next_data = None
        self.preload()

    def preload(self):
        try:
            self.next_data = TensorTuple(next(self.loader))
        except StopIteration:
            self.next_data = None
            raise StopIteration
        if self._cuda_available:
            with torch.cuda.stream(self.stream):
                self.next_data = self.next_data.to(device="cuda", non_blocking=True)

    def __len__(self):
        return self._length

    def __iter__(self):
        return self

    def __next__(self):
        if self._cuda_available:
            torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self.preload()
        return data
