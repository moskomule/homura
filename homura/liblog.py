""" logging tools leaned a lot from Optuna and Transformers
"""
import io
import logging
import sys
import threading
import warnings
from typing import TextIO

import tqdm as _tqdm
from tqdm.contrib import DummyTqdmFile

try:
    import colorlog

    _has_colorlog = True
except ImportError:
    _has_colorlog = False

# private APIs
_LOG_LEVEL = {"debug": logging.DEBUG,
              "info": logging.INFO,
              "warning": logging.WARNING,
              "error": logging.ERROR,
              "critical": logging.CRITICAL}

_default_handler = None
_original_stds = sys.stdout, sys.stderr
_lock = threading.Lock()


def _name() -> str:
    return __name__.split('.')[0]


def _create_default_formatter() -> logging.Formatter:
    datefmt = "%Y-%m-%d %H:%M:%S"
    return colorlog.ColoredFormatter('%(log_color)s[%(name)s|%(asctime)s|%(levelname)s] %(message)s', datefmt=datefmt)


def _get_root_logger() -> logging.Logger:
    return logging.getLogger(_name())


def _configure_root_logger() -> None:
    global _default_handler
    with _lock:
        if _default_handler is not None:
            return None
        _default_handler = logging.StreamHandler()
        _default_handler.setFormatter(_create_default_formatter())
        _user_root_logger = logging.getLogger()
        if len(_user_root_logger.handlers) > 0:
            # if user already defines their own root logger
            return None
        root_logger = _get_root_logger()
        root_logger.addHandler(_default_handler)
        root_logger.setLevel(logging.INFO)
        root_logger.propagate = False


def _reset_root_logger() -> None:
    global _default_handler
    with _lock:
        if _default_handler is None:
            return None
        root_logger = _get_root_logger()
        root_logger.removeHandler(_default_handler)
        root_logger.setLevel(logging.NOTSET)
        _default_handler = None


# public APIs

def get_logger(name: str = None
               ) -> logging.Logger:
    if name is None:
        name = _name()
    _configure_root_logger()
    return logging.getLogger(name)


def get_verb_level() -> int:
    _configure_root_logger()
    return _get_root_logger().getEffectiveLevel()


def set_verb_level(level: str or int) -> None:
    if isinstance(level, str):
        level = _LOG_LEVEL[level]
    _configure_root_logger()
    _get_root_logger().setLevel(level)


def enable_default_handler() -> None:
    _configure_root_logger()
    if _default_handler is None:
        raise RuntimeWarning()
    _get_root_logger().addHandler(_default_handler)


def disable_default_handler() -> None:
    _configure_root_logger()
    if _default_handler is None:
        raise RuntimeWarning()
    _get_root_logger().removeHandler(_default_handler)


def enable_propagation() -> None:
    _configure_root_logger()
    _get_root_logger().propagate = True


def disable_propagation() -> None:
    _configure_root_logger()
    _get_root_logger().propagate = False


def set_file_handler(log_file: str or TextIO, level: str or int = logging.DEBUG,
                     formatter: logging.Formatter = None) -> None:
    _configure_root_logger()
    fh = logging.FileHandler(log_file)
    if isinstance(level, str):
        level = _LOG_LEVEL[level]
    fh.setLevel(level)
    if formatter is None:
        formatter = _create_default_formatter()
    fh.setFormatter(formatter)
    _get_root_logger().addHandler(fh)


# internal APIs
def set_tqdm_handler(level: str or int = logging.INFO,
                     formatter: logging.Formatter = None) -> None:
    """ An alternative handler to avoid disturbing tqdm
    """

    import tqdm

    class TQDMHandler(logging.StreamHandler):
        def __init__(self):
            logging.StreamHandler.__init__(self)

        def emit(self, record):
            msg = self.format(record)
            tqdm.tqdm.write(msg)

    _configure_root_logger()
    th = TQDMHandler()
    if isinstance(level, str):
        level = _LOG_LEVEL[level]
    th.setLevel(level)
    if _default_handler is not None:
        # to avoid multiple logs!
        _get_root_logger().removeHandler(_default_handler)
    if formatter is None:
        formatter = _create_default_formatter()
    th.setFormatter(formatter)
    _get_root_logger().addHandler(th)


# tqdm

def set_tqdm_stdout_stderr():
    # https://github.com/tqdm/tqdm/blob/master/examples/redirect_print.py
    # Some libraries override sys.stdout, which causes OSError: [Errno 9] Bad file descriptor.
    # To avoid this, this if statement is necessary
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout, sys.stderr = map(DummyTqdmFile, _original_stds)
    elif not isinstance(sys.stdout, DummyTqdmFile):
        warnings.warn(f"sys.stdout is unexpected type: {type(sys.stdout)}.\n"
                      f"If you use wandb, set WANDB_CONSOLE=off to avoid tqdm-related problems.",
                      UserWarning)


def tqdm(*args, **kwargs):
    # https://github.com/tqdm/tqdm/blob/master/examples/redirect_print.py
    if kwargs.get("file") is None:
        kwargs["file"] = _original_stds[0]
        # tqdm seems to prioritize dynamic_ncols over ncols
    if kwargs.get("ncols") is None and kwargs.get("dynamic_ncols") is None:
        kwargs["dynamic_ncols"] = True
    return _tqdm.tqdm(*args, **kwargs)


# log once
_LOG_CACHE = set()


def log_once(logger,
             message: str,
             key=str) -> None:
    """ Log message only once.

    :param logger: e.g., `print`, `logger.info`
    :param message:
    :param key: if `key=None`, `message` is used as `key`.
    :return:
    """

    if key is None:
        key = message
    if key in _LOG_CACHE:
        return
    logger(message)
    _LOG_CACHE.add(key)


def print_once(message: str,
               key=str) -> None:
    """ `print` version of `log_once`
    """

    log_once(print, message, key)
