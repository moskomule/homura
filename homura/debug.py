from typing import Optional, Callable, Tuple

import torch
from torch import nn

from homura.liblog import get_logger, set_verb_level

logger = get_logger(__name__)
set_verb_level("debug")


def _format(tt: Tuple[torch.Tensor]):
    s = ""
    for t in tt:
        if torch.is_tensor(t):
            s += f"Tensor(shape={tuple(t.size())}), "
        else:
            s += f"{type(t)}, "
    return s.strip(", ")


def _forward_log(m: nn.Module, input: Tuple[torch.Tensor], output: Tuple[torch.Tensor]) -> None:
    logger.debug(f">>forward: name={m.__class__.__name__} input={_format(input)} output={_format(output)}")


def _backward_log(m: nn.Module, input: Tuple[torch.Tensor], output: Tuple[torch.Tensor]) -> None:
    logger.debug(f">>backward: name={m.__class__.__name__} input={_format(input)} output={_format(output)}")


def simple_debugger(model: nn.Module, input: torch.Tensor,
                    target: Optional[torch.Tensor] = None, loss: Optional[Callable] = None) -> None:
    if target is not None and loss is None:
        raise TypeError(f"argument loss should be Callable but got None")
    if isinstance(model, nn.DataParallel) or isinstance(model, nn.parallel.DistributedDataParallel):
        logger.warning(f"Debugger may not be able to work with {type(model)}")
    model.apply(lambda m: m.register_forward_hook(_forward_log))
    model.apply(lambda m: m.register_backward_hook(_backward_log))
    logger.debug("Start forward calculation")
    output = model(input)
    if loss is not None:
        output = loss(output, target)
    else:
        output = output.mean()
    logger.debug("Start backward calculation")
    output.backward()
