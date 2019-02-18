import importlib.util

is_apex_available = False
if importlib.util.find_spec("apex") is not None:
    is_apex_available = True

is_tensorboard_available = False
if importlib.util.find_spec("tensorboardX") is not None:
    is_tensorboard_available = True
