from pathlib import Path
import yaml
import json
import argparse


class HyperParameter(object):
    def __init__(self, *args, **kwargs):
        """
        Hyper parameter administrator
        >>> pa = HyperParameter("config.json", lr=0.1)
        """
        self._container = {}
        if len(args) > 0:
            self._load_files(args)

        if len(kwargs) > 0:
            self._check_duplicated(kwargs.keys())
            self._update(kwargs)
        self._set_getter()

    def register_hp(self, *, file_name=None, args=None):
        if file_name is not None:
            self._load_file(Path(file_name))
        elif isinstance(args, argparse.Namespace):
            dic = vars(args)
            self._check_duplicated(dic.keys())
            self._update(dic)
        else:
            raise Exception("Unknown argument!")
        self._set_getter()

    def _load_files(self, args: tuple):
        for path in [Path(p) for p in args]:
            self._load_file(path)

    def _load_file(self, path: Path):
        with open(path) as f:
            if path.suffix == ".json":
                dic = json.load(f)
            elif path.suffix == ".yaml":
                dic = yaml.load(f)
            else:
                raise Exception(f"Unknown file type {path.stem}")
        self._check_duplicated(dic.keys())
        self._update(dic)

    def _check_duplicated(self, keys):
        for k in self._container.keys():
            if k in keys:
                raise Exception(f"key {k} is already used!")

    def _update(self, dic: dict):
        _dic = {}
        for k, v in dic.items():
            if isinstance(v, dict):
                v = HyperParameter(**v)
            _dic[k] = v
        self._container.update(_dic)

    def _set_getter(self):
        # todo: these values are mutable so find a way to use setter
        for k, v in self._container.items():
            setattr(self, k, v)

    def __str__(self):
        string = "Hyper Parameters: \n"
        for k, v in self._container.items():
            string += f"{k}: {v}\n"
        return string
