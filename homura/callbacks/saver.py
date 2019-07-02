from collections.abc import Mapping
from pathlib import Path

import torch

from .base import Callback, _NoOpCallback
from ..utils._vocabulary import *
from ..utils.environment import get_global_rank, get_git_hash, get_args


class WeightSave(Callback):
    """ Save weights after every epoch

    :param save_path: path to be saved
    :param save_freq: frequency of saving in epoch. If -1, saved by `after_all`.
    """

    def __new__(cls, *args, **kwargs):
        if get_global_rank() > 0:
            return _NoOpCallback()
        else:
            return object.__new__(cls)

    def __init__(self,
                 save_path: str or Path,
                 save_freq: int = 1):

        postfix = ""
        if len(get_git_hash()) > 0:
            postfix = "-" + get_git_hash()
        self.save_path = Path(save_path) / (BASIC_DIR_NAME + postfix)
        self.save_freq = save_freq
        self._epoch = 0
        self._step = 0

        if not self.save_path.exists():
            self.save_path.mkdir(parents=True)

    def save(self,
             data: Mapping,
             file_name: str):
        try:
            # scheduler is not a must
            scheduler_state_dict = data.get(SCHEDULER)
            if scheduler_state_dict is not None:
                scheduler_state_dict = scheduler_state_dict.state_dict()
                
            torch.save({"git": get_git_hash(),
                        "args": get_args(),
                        MODEL: data[MODEL].state_dict(),
                        OPTIMIZER: data[OPTIMIZER].state_dict(),
                        SCHEDULER: scheduler_state_dict,
                        EPOCH: self._epoch,
                        STEP: self._step},
                       self.save_path / file_name)
        except Exception as e:
            raise e

    def after_epoch(self, data: Mapping):
        self._epoch = data[EPOCH]
        self._step = data[STEP]
        if self.save_freq > 0 and data[EPOCH] % self.save_freq == 0:
            self.save(data, f"{data[EPOCH]}.pkl")

    def after_all(self, data: Mapping):
        if self.save_freq == -1:
            self.save(data, "weight.pkl")
