import contextlib
import functools
import types
from pathlib import Path
from typing import Callable, Optional, Dict, Type, Any


class Registry(object):
    """ Registry of models, datasets and anything you like. ::

        model_registry = Registry('model')
        @model_registry.register
        def your_model(*args, **kwargs):
            return ...
        your_model_instance = model_registry('your_model')(...)
        model_registry2 = Registry('model')
        model_registry is model_registry2
    """
    _available_registries = {}

    def __new__(cls,
                name: str,
                type: Optional[Type] = None):
        if name in Registry._available_registries:
            return Registry._available_registries[name]

        return object.__new__(cls)

    def __init__(self,
                 name: str,
                 type: Optional[Type] = None):
        self.name = name
        Registry._available_registries[name] = self
        self.type = type
        self._registry = {}

    def register_from_dict(self,
                           name_to_func: Dict[str, Callable]):
        for k, v in name_to_func.items():
            self.register(v, name=k)

    def register(self,
                 func: Callable = None,
                 *,
                 name: Optional[str] = None):
        if func is None:
            return functools.partial(self.register, name=name)

        _type = self.type
        if _type is not None and not isinstance(_type, types.FunctionType):
            if not (isinstance(func, _type) or issubclass(func, _type)):
                raise TypeError(
                    f'`func` is expected to be subclass of {_type}.')

        if name is None:
            name = func.__name__

        if self._registry.get(name) is not None:
            raise KeyError(
                f'Name {name} is already used, try another name!')

        self._registry[name] = func
        return func

    def __call__(self,
                 name: str):
        ret = self._registry.get(name)
        if ret is None:
            _registry = {k.lower(): v for k, v in self._registry.items()}
            ret = _registry.get(name.lower())
            if ret is None:
                raise KeyError(f'Unknown {name} is called!')
        return ret

    @classmethod
    def available_registries(cls,
                             detailed: bool = False):
        if detailed:
            for k, v in cls._available_registries.items():
                v.catalogue()
        else:
            print(list(cls._available_registries.keys()))

    @classmethod
    def help(cls):
        cls.available_registries(detailed=True)

    def catalogue(self):
        print(f"{self.name} {list(self._registry.keys())}")

    @staticmethod
    def import_modules(package_name: str) -> None:
        import importlib, pkgutil

        importlib.invalidate_caches()
        with Registry._push_python_path("."):
            module = importlib.import_module(package_name)
            path = getattr(module, '__path__', [])
            path_string = "" if not path else path[0]

            for module_finder, name, _ in pkgutil.walk_packages(path):
                if path_string and module_finder.path != path_string:
                    continue
                sub_package = f'{package_name}.{name}'
                Registry.import_modules(sub_package)

    @staticmethod
    @contextlib.contextmanager
    def _push_python_path(path: str):
        # https://github.com/allenai/allennlp/blob/v1.0.0/allennlp/common/util.py
        import sys

        path = Path(path).resolve()
        sys.path.insert(0, str(path))
        try:
            yield
        finally:
            sys.path.remove(str(path))
