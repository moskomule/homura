import torch
from torch.utils.data import DataLoader, DistributedSampler

from homura import TensorTuple


class DataPrefetchWrapper(object):
    """ from NVidia's DeepLearningExamples ::

    """

    def __init__(self,
                 loader: DataLoader,
                 start_epoch: int = 0
                 ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError('Prefetcher needs CUDA, but not available!')
        self.loader = loader
        self.epoch = start_epoch - 1

    def __iter__(self):
        if self.loader.sampler is not None and isinstance(self.loader.sampler, DistributedSampler):
            self.loader.sampler.set_epoch(self.epoch)
        self.epoch += 1

        stream = torch.cuda.Stream()
        is_first = True

        for next_data in self.loader:
            with torch.cuda.stream(stream):
                next_data = TensorTuple(next_data).to(device='cuda', non_blocking=True)

            if not is_first:
                yield data
            else:
                is_first = False

            torch.cuda.current_stream().wait_stream(stream)
            data = next_data
        yield data

    def __len__(self) -> int:
        return len(self.loader)
