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
                registry_name: str,
                type: Type[Any] = None):
        # if registry_name is already used,
        # return the corresponding registry
        registry_name = registry_name.lower()
        if registry_name in Registry._available_registries:
            return Registry._available_registries[registry_name]

        # else create new one
        return object.__new__(cls)

    def __init__(self,
                 registry_name: str,
                 type: Type[Any] = None):
        self.name = registry_name.lower()
        Registry._available_registries[self.name] = self
        self.type = type
        self._registry = {}

    def register_from_dict(self,
                           name_to_func: Dict):
        for k, v in name_to_func.items():
            self.register(v, entry_name=k)

    def register(self,
                 func: Callable = None,
                 *,
                 entry_name: Optional[str] = None):
        if func is None:
            return functools.partial(self.register, name=entry_name)

        _type = self.type
        if _type is not None and not isinstance(_type, types.FunctionType):
            if not (isinstance(func, _type) or issubclass(func, _type)):
                raise TypeError(
                    f'`func` is expected to be subclass of {_type}.')

        if entry_name is None:
            entry_name = func.__name__
        entry_name = entry_name.lower()

        if self._registry.get(entry_name) is not None:
            raise KeyError(
                f'Name {entry_name} is already used, try another name!')

        self._registry[entry_name] = func
        return func

    def __call__(self,
                 name: str):
        ret = self._registry.get(name.lower())
        if ret is None:
            raise KeyError(f'Unknown {name} is called!')
        return ret

    @classmethod
    def available_registries(cls,
                             detailed: bool = False):
        # todo: table style
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
        """ Import all submodules under the given package and make all registries ready.
        """

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
