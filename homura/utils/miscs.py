from torch.autograd import Variable
import subprocess


def _decode_bytes(b: bytes):
    return b.decode("ascii")[:-1]


def get_git_hash():
    is_git_repo = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"],
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout
    if _decode_bytes(is_git_repo) == "true":
        git_hash = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                  stdout=subprocess.PIPE).stdout
        return _decode_bytes(git_hash)
    else:
        return ""


def to_tensor(v, cpu=True):
    if isinstance(v, Variable):
        v = v.data
    if cpu:
        v = v.cpu()
    return v


if __name__ == '__main__':
    get_git_hash()
