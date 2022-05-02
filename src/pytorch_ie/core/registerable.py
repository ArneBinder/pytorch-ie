import inspect
from collections import defaultdict
from typing import Dict, Optional, Type, TypeVar


class RegistrationError(Exception):
    pass


T = TypeVar("T", bound="Registrable")


class Registrable:
    _registry: Dict[Type, Dict[str, Type]] = defaultdict(dict)
    _register_name: Optional[str] = None

    @classmethod
    def register(
        cls: Type[T],
        name: Optional[str] = None,
    ):
        registry = Registrable._registry[cls]

        def add_subclass_to_registry(subclass: Type[T]) -> Type[T]:

            # save the name
            subclass._register_name = name

            # ignore type because mypy has an issue with class properties,
            # see https://github.com/python/mypy/issues/2563
            register_name: str = subclass.register_name  # type: ignore

            if not inspect.isclass(subclass) or not issubclass(subclass, cls):
                raise RegistrationError(
                    f"Cannot register {subclass.__name__} as {register_name}; "
                    f"{subclass.__name__} must be a subclass of {cls.__name__}"
                )

            if register_name in registry:
                raise RegistrationError(
                    f"Cannot register {subclass.__name__} as {register_name}; "
                    f"name already in use for {registry[register_name].__name__}"
                )

            registry[register_name] = subclass
            return subclass

        return add_subclass_to_registry

    @classmethod
    @property
    def register_name(cls: Type[T]) -> str:
        # default to class name
        return cls.__name__ if cls._register_name is None else cls._register_name

    @classmethod
    def by_name(cls: Type[T], name: str) -> Type[T]:
        if name in Registrable._registry[cls]:
            return Registrable._registry[cls][name]

        raise RegistrationError(f"{name} is not a registered name for {cls.__name__}.")
