import importlib.util

is_apex_available = importlib.util.find_spec("apex") is not None

is_tensorboardX_available = importlib.util.find_spec("tensorboardX") is not None

is_accimage_available = importlib.util.find_spec("accimage") is not None
