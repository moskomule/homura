from pathlib import Path
from types import MethodType
from typing import Dict, Iterable, Tuple, Callable

import torch
from homura.callbacks import Callback
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .containers import Map
from .miscs import check_path
from .runner import Runner
from ._vocabulary import *


class Inferencer(Runner):
    """ Runner for inference only.
    """

    mode = "inference"

    def __init__(self, model: nn.Module or Dict[str, nn.Module],
                 callbacks: Callback or Iterable[Callback] = None,
                 device: torch.device or str = None,
                 verb=True, use_cudnn_benchmark=True, use_cuda_nonblocking=False, **kwargs):
        """
        Runner for inference
        :param model:
        :param callbacks:
        :param device:
        :param verb:
        :param use_cudnn_benchmark:
        :param use_cuda_nonblocking:
        :param kwargs:
        """
        super(Inferencer, self).__init__(model, callbacks, device, use_cudnn_benchmark, use_cuda_nonblocking, **kwargs)
        self.model.eval()
        self._verb = verb
        self._is_model_loaded = False
        # to be compatible with iteration in Trainer
        self.is_train = False
        self.loss_f = lambda *x: 0

    def _iteration(self, data: Tuple[torch.Tensor]) -> Map:
        if not self._is_model_loaded:
            raise RuntimeError("model is not loaded yet")
        with torch.no_grad():
            output = self.iteration(data)
            if not isinstance(output, dict):
                output = Map(output=output)
            return output

    def load(self, path: str or Path):
        path = check_path(path)
        with path.open('rb') as f:
            loaded = torch.load(f)
        self.model.load_state_dict(loaded[MODEL])
        self._is_model_loaded = True

    def iteration(self, data: Tuple[torch.Tensor]) -> Map:
        input = data[0].to(self.device)
        return Map(output=self.model(input))

    def override_iteration(self, new_iteration: Callable):
        setattr(self, "iteration", MethodType(new_iteration, self))

    def update_loss_f(self, loss_f: Callable):
        self.loss_f = loss_f

    def run(self, data_loader: Iterable[torch.Tensor]):
        self._callbacks.before_all(dict(mode=self.mode))
        cycle_map = {ITER_PER_EPOCH: 0, EPOCH: 0, MODE: self.mode}
        self._callbacks.before_epoch(cycle_map)
        if self._verb:
            data_loader = tqdm(data_loader, ncols=80)
        for step, data in enumerate(data_loader):
            iter_map = Map(mode=self.mode, step=step)
            self._callbacks.before_iteration(iter_map)
            output_map = self._iteration(data)
            output_map[DATA] = data
            output_map.update(iter_map)
            self._callbacks.after_iteration(output_map)
            cycle_map[ITER_PER_EPOCH] = step + 1
        self._callbacks.after_epoch(cycle_map)
        self._callbacks.after_all(dict(mode=self.mode))

    def test(self, data_loader: DataLoader):
        # compatible with Trainer
        self.run(data_loader)
