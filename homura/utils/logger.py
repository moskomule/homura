import logging
from typing import TextIO

__all__ = ["get_logger"]

LOG_LEVEL = {"debug": logging.DEBUG,
             "info": logging.INFO,
             "warning": logging.WARNING,
             "error": logging.ERROR,
             "critical": logging.CRITICAL}


def get_logger(name: str = None, log_file: str or TextIO = None, *,
               file_filter_level: str = "debug", stdout_filter_level: str = "debug"):
    """
    basic logger
    :param name: name of logger
    :param log_file:
    :param file_filter_level:
    :param stdout_filter_level:
    :return:
    """
    name = __name__ if name is None else name
    logger = logging.getLogger(name=name)
    formatter = logging.Formatter("[%(name)s|%(asctime)s|%(levelname)s] %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")

    if log_file is not None:
        fh = logging.FileHandler(log_file)
        if file_filter_level not in LOG_LEVEL.keys():
            raise ValueError(f"{file_filter_level} is not a correct log level! ({LOG_LEVEL.keys()})")
        file_filter_level = LOG_LEVEL[file_filter_level]
        fh.setLevel(file_filter_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    ch = logging.StreamHandler()

    if stdout_filter_level not in LOG_LEVEL.keys():
        raise ValueError(f"{stdout_filter_level} is not a correct log level! ({LOG_LEVEL.keys()})")
    stdout_filter_level = LOG_LEVEL[stdout_filter_level]

    ch.setLevel(stdout_filter_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger
