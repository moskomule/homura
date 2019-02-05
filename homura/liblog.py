# ref to optuna logging

import logging
from typing import Optional, TextIO

# private APIs
_LOG_LEVEL = {"debug": logging.DEBUG,
              "info": logging.INFO,
              "warning": logging.WARNING,
              "error": logging.ERROR,
              "critical": logging.CRITICAL}

_default_handler = None


def _name() -> str:
    return __name__.split('.')[0]


def _create_default_formatter() -> logging.Formatter:
    return logging.Formatter("[%(name)s|%(asctime)s|%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def _get_root_logger() -> logging.Logger:
    return logging.getLogger(_name())


def _configure_root_logger() -> None:
    global _default_handler
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


def _reset_root_logger() -> None:
    global _default_handler
    if _default_handler is None:
        return None
    root_logger = _get_root_logger()
    root_logger.removeHandler(_default_handler)
    root_logger.setLevel(logging.NOTSET)
    _default_handler = None


# public APIs

def get_logger(name: str):
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
    if _default_handler is not None:
        raise RuntimeWarning()
    _get_root_logger().removeHandler(_default_handler)


def set_file_handler(log_file: str or TextIO, level: str or int = logging.DEBUG,
                     formatter: Optional[logging.Formatter] = None) -> None:
    _configure_root_logger()
    fh = logging.FileHandler(log_file)
    if isinstance(level, str):
        level = _LOG_LEVEL[level]
    fh.setLevel(level)
    if formatter is None:
        formatter = _create_default_formatter()
    fh.setFormatter(formatter)
    _get_root_logger().addHandler(fh)
