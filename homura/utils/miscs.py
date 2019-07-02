from pathlib import Path


def check_path(path: str or Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{str(path)} not exists!")
    return path
