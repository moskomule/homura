from abc import ABCMeta
from pathlib import Path

import torch
from omegaconf import OmegaConf

from .trainers import TrainerBase


class TaskBase(metaclass=ABCMeta):

    def __init__(self,
                 config: OmegaConf):
        self._config = config
        self._trainer = TrainerBase
        self.trainer = None

    def _init_trainer(self):
        self.trainer = self._trainer(self._get_model(),
                                     self._get_optimizer(),
                                     self._get_loss_f())

    def _get_model(self):
        ...

    def _get_optimizer(self):
        ...

    def _get_loaders(self):
        ...

    def _get_scheduler(self):
        ...

    def _get_loss_f(self):
        ...

    def run(self):
        ...

    def tune(self):
        ...

    def eval(self):
        ...

    def save(self):
        ...

    def _load(self,
              path: Path or str):
        path = Path(path)
        try:
            _loaded = torch.load(path)

        except Exception as e:
            raise e

    def resume(self):
        ...
