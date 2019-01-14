import subprocess
from pathlib import Path


def _decode_bytes(b: bytes):
    return b.decode("ascii")[:-1]


def get_git_hash():
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
        return ""


def check_path(path: str or Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{str(path)} not exists!")
    return path
