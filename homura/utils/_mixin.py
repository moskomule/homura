from pathlib import Path
from typing import Any, Dict, Optional

import torch

from homura import if_is_master


class StateDictMixIn(object):
    def state_dict(self
                   ) -> Dict[str, Any]:
        raise NotImplementedError

    def load_state_dict(self,
                        state_dict: Dict[str, Any]
                        ) -> None:
        raise NotImplementedError

    @if_is_master
    def save(self,
             path: str,
             file_name: str
             ) -> None:
        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)
        with (path / f"{file_name}.pt").open("wb") as f:
            torch.save(self.state_dict(), f)
            if hasattr(self, 'logger'):
                self.logger.info(f"Weight saved! ({file_name})")

    def load(self,
             path: str,
             file_name: Optional[str],
             device: Optional[torch.device] = None
             ) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Path {path} not found!")
        if file_name is None:
            paths = list(path.glob("*.pt"))
            if len(paths) == 0:
                raise FileNotFoundError(f"Path {path} does not contain checkpoints")
            path = sorted(paths, key=lambda x: int(x.stem))[-1]
        else:
            path = path / file_name

        with path.open('rb') as f:
            state_dict = torch.load(f, map_location=device)

        self.load_state_dict(state_dict)
        if hasattr(self, 'logger'):
            self.logger.info(f"Weight {path} is successfully loaded!")
