import inspect
from collections import defaultdict
from typing import Dict, Optional, Type, TypeVar


class RegistrationError(Exception):
    pass


T = TypeVar("T", bound="Registrable")


class Registrable:
    _registry: Dict[Type, Dict[str, Type]] = defaultdict(dict)

    @classmethod
    def register(
        cls: Type[T],
        name: Optional[str] = None,
    ):
        registry = Registrable._registry[cls]

        def add_subclass_to_registry(subclass: Type[T]) -> Type[T]:
            register_name = subclass.__name__ if name is None else name

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
    def by_name(cls: Type[T], name: str) -> Type[T]:
        if name in Registrable._registry[cls]:
            return Registrable._registry[cls][name]

        raise RegistrationError(f"{name} is not a registered name for {cls.__name__}.")

    @classmethod
    def registered_name_for_class(cls: Type[T], clazz: Type[T]) -> Optional[str]:
        inverse_lookup = {v: k for k, v in Registrable._registry[cls].items()}
        return inverse_lookup.get(clazz)
