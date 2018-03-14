import subprocess


def get_git_hash():
    return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"]).decode("ascii")[:-1]
