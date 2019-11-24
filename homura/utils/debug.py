from collections import Counter
from functools import partial
from typing import Optional, Callable, Tuple

import torch
from torch import nn

from homura.liblog import get_logger, set_verb_level, get_verb_level

__all__ = ["module_debugger"]

logger = get_logger(__name__)
_counter = Counter()


def _extend_apply(self: nn.Module, fn: Callable):
    """
    extend nn.Module.apply
    """
    if not hasattr(self, "debug_depth"):
        self.debug_depth = 0
    self.debug_id = _counter[self.__class__.__name__]
    _counter[self.__class__.__name__] += 1

    for module in self.children():
        module.debug_depth = self.debug_depth + 1
        module.extend_apply(fn)
    fn(self)
    return self


def _log(message: str, m: nn.Module, *_) -> None:
    logger.debug(f"{message}>{'  ' * m.debug_depth} name={m.__class__.__name__}({m.debug_id})")


def module_debugger(model: nn.Module,
                    input: Tuple[torch.Tensor] or torch.Tensor,
                    target: Optional[Tuple[torch.Tensor]] = None,
                    loss: Optional[Callable] = None) -> None:
    """ log all modules connected with forward and backward calculation
    """

    original_verb = get_verb_level()
    set_verb_level("debug")
    nn.Module.extend_apply = _extend_apply
    if target is not None and loss is None:
        raise TypeError(f"Argument loss should be Callable but got None")
    if isinstance(model, nn.DataParallel) or isinstance(model, nn.parallel.DistributedDataParallel):
        logger.warning(f"Debugger may not be able to work with {type(model)}")
    model.extend_apply(lambda m: m.register_forward_pre_hook(partial(_log, "forward")))
    model.extend_apply(lambda m: m.register_backward_hook(partial(_log, "backward")))
    logger.info("Start debugging mode")
    logger.debug("Start forward calculation")
    if torch.is_tensor(input):
        input = (input,)
    output = model(*input)
    if loss is not None:
        output = loss(output, target)
    else:
        output = output.mean()
    logger.debug("Start backward calculation")
    output.backward()
    logger.info("Finish debugging mode")
    set_verb_level(original_verb)
