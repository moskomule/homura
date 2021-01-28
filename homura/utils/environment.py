""" Helper functions to get information about the environment.
"""

import importlib.util
import os as python_os
import subprocess
import sys as python_sys
from typing import Any, Optional

from homura.liblog import get_logger

logger = get_logger("homura.environment")


# Utility functions that useful libraries are available or not
def is_accimage_available() -> bool:
    return importlib.util.find_spec("accimage") is not None


def enable_accimage() -> None:
    if is_accimage_available():
        import torchvision

        torchvision.set_image_backend("accimage")
        logger.info("accimage is activated")
    else:
        logger.warning("accimage is not available")


def is_faiss_available() -> bool:
    _faiss_available = importlib.util.find_spec("faiss") is not None
    if _faiss_available:
        import faiss
        if not hasattr(faiss, 'StandardGpuResources'):
            logger.info("faiss is available but is not for GPUs")
    return _faiss_available


def is_cupy_available() -> bool:
    return importlib.util.find_spec("cupy") is not None


def is_opteinsum_available() -> bool:
    return importlib.util.find_spec("opt_einsum") is not None


# get environment information

def get_git_hash() -> str:
    def _decode_bytes(b: bytes) -> str:
        return b.decode("ascii")[:-1]

    try:
        is_git_repo = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"],
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout
    except FileNotFoundError:
        return ""

    if _decode_bytes(is_git_repo) == "true":
        git_hash = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                  stdout=subprocess.PIPE).stdout
        return _decode_bytes(git_hash)
    else:
        logger.info("No git info available in this directory")
        return ""


def get_args() -> list:
    return python_sys.argv


def get_environ(name: str,
                default: Optional[Any] = None
                ) -> str:
    return python_os.environ.get(name, default)
