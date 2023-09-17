# taken from hydra/_internal/instantiate/_instantiate2.py
from typing import Any, Callable, Optional, Type, Union

from pytorch_ie.core.document import Document


class HydraException(Exception):
    ...


class CompactHydraException(HydraException):
    ...


class InstantiationException(CompactHydraException):
    ...


def _locate(path: str) -> Any:
    """Locate an object by name or dotted path, importing as necessary.

    This is similar to the pydoc function `locate`, except that it checks for
    the module from the given path from back to front.
    """
    if path == "":
        raise ImportError("Empty path")
    from importlib import import_module
    from types import ModuleType

    parts = [part for part in path.split(".")]
    for part in parts:
        if not len(part):
            raise ValueError(
                f"Error loading '{path}': invalid dotstring."
                + "\nRelative imports are not supported."
            )
    assert len(parts) > 0
    part0 = parts[0]
    try:
        obj = import_module(part0)
    except Exception as exc_import:
        raise ImportError(
            f"Error loading '{path}':\n{repr(exc_import)}"
            + f"\nAre you sure that module '{part0}' is installed?"
        ) from exc_import
    for m in range(1, len(parts)):
        part = parts[m]
        try:
            obj = getattr(obj, part)
        except AttributeError as exc_attr:
            parent_dotpath = ".".join(parts[:m])
            if isinstance(obj, ModuleType):
                mod = ".".join(parts[: m + 1])
                try:
                    obj = import_module(mod)
                    continue
                except ModuleNotFoundError as exc_import:
                    raise ImportError(
                        f"Error loading '{path}':\n{repr(exc_import)}"
                        + f"\nAre you sure that '{part}' is importable from module '{parent_dotpath}'?"
                    ) from exc_import
                except Exception as exc_import:
                    raise ImportError(
                        f"Error loading '{path}':\n{repr(exc_import)}"
                    ) from exc_import
            raise ImportError(
                f"Error loading '{path}':\n{repr(exc_attr)}"
                + f"\nAre you sure that '{part}' is an attribute of '{parent_dotpath}'?"
            ) from exc_attr
    return obj


def resolve_target(
    target: Union[str, type, Callable[..., Any]], full_key: str = ""
) -> Union[type, Callable[..., Any]]:
    """Resolve target string, type or callable into type or callable."""
    if isinstance(target, str):
        try:
            target = _locate(target)
        except Exception as e:
            msg = f"Error locating target '{target}', set env var HYDRA_FULL_ERROR=1 to see chained exception."
            if full_key:
                msg += f"\nfull_key: {full_key}"
            raise InstantiationException(msg) from e
    if not callable(target):
        msg = f"Expected a callable target, got '{target}' of type '{type(target).__name__}'"
        if full_key:
            msg += f"\nfull_key: {full_key}"
        raise InstantiationException(msg)
    return target


def resolve_optional_document_type(
    document_type: Optional[Union[str, Type[Document]]]
) -> Optional[Type[Document]]:
    if document_type is None:
        return None
    if isinstance(document_type, str):
        dt = resolve_target(document_type)
    else:
        dt = document_type
    if not (isinstance(dt, type) and issubclass(dt, Document)):
        raise TypeError(
            f"(resolved) document_type must be a subclass of Document, but it is: {dt}"
        )
    return dt


def serialize_document_type(document_type: Type[Document]) -> str:
    return f"{document_type.__module__}.{document_type.__name__}"
