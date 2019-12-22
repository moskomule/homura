from collections.abc import Mapping
from pathlib import Path

import torch

from .base import Callback
from ..utils._vocabulary import *
from ..utils.environment import get_git_hash, get_args

Path.remove = lambda self: os.remove(self)


class WeightSave(Callback):
    """ Save weights after every epoch

    :param save_path: path to be saved
    :param save_freq: frequency of saving in epoch. If -1, saved by `after_all`.
    """
    master_only = True

    def __init__(self,
                 save_path: str or Path,
                 last_only: bool = False):

        self.save_path = Path(save_path) / UNIQUE_ID
        self.last_only = last_only

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
                        EPOCH: data[EPOCH],
                        ITERATION: data[ITERATION]},
                       self.save_path / file_name)
        except Exception as e:
            raise e

    def delete(self,
               file_name: str):
        file = self.save_path / file_name
        if file.exists():
            file.remove()

    def after_epoch(self, data: Mapping):
        self.save(data, f"{data[EPOCH]}.pkl")
        if self.last_only:
            self.delete(f"{data[EPOCH] - 1}.pkl")

    def after_all(self, data: Mapping):
        self.save(data, "weight.pkl")
