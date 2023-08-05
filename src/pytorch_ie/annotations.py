from dataclasses import dataclass, field
from typing import Optional, Tuple

from pytorch_ie.core.document import Annotation


def _post_init_single_label(self):
    if not isinstance(self.label, str):
        raise ValueError("label must be a single string.")

    if not isinstance(self.score, float):
        raise ValueError("score must be a single float.")


def _post_init_multi_label(self):
    if self.score is None:
        score = tuple([1.0] * len(self.label))
        object.__setattr__(self, "score", score)

    if not isinstance(self.label, tuple):
        object.__setattr__(self, "label", tuple(self.label))

    if not isinstance(self.score, tuple):
        object.__setattr__(self, "score", tuple(self.score))

    if len(self.label) != len(self.score):
        raise ValueError(
            f"Number of labels ({len(self.label)}) and scores ({len(self.score)}) must be equal."
        )


def _post_init_multi_span(self):
    if isinstance(self.slices, list):
        object.__setattr__(self, "slices", tuple(tuple(s) for s in self.slices))


def _post_init_arguments_and_roles(self):
    if len(self.arguments) != len(self.roles):
        raise ValueError(
            f"Number of arguments ({len(self.arguments)}) and roles ({len(self.roles)}) must be equal"
        )
    if not isinstance(self.arguments, tuple):
        object.__setattr__(self, "arguments", tuple(self.arguments))
    if not isinstance(self.roles, tuple):
        object.__setattr__(self, "roles", tuple(self.roles))


@dataclass(eq=True, frozen=True)
class Label(Annotation):
    label: str
    score: float = field(default=1.0, compare=False)

    def __post_init__(self) -> None:
        _post_init_single_label(self)


@dataclass(eq=True, frozen=True)
class MultiLabel(Annotation):
    label: Tuple[str, ...]
    score: Optional[Tuple[float, ...]] = field(default=None, compare=False)

    def __post_init__(self) -> None:
        _post_init_multi_label(self)


@dataclass(eq=True, frozen=True)
class Span(Annotation):
    start: int
    end: int

    def __str__(self) -> str:
        if self.target is None:
            return ""
        return str(self.target[self.start : self.end])


@dataclass(eq=True, frozen=True)
class LabeledSpan(Span):
    label: str
    score: float = field(default=1.0, compare=False)

    def __post_init__(self) -> None:
        _post_init_single_label(self)


@dataclass(eq=True, frozen=True)
class MultiLabeledSpan(Span):
    label: Tuple[str, ...]
    score: Optional[Tuple[float, ...]] = field(default=None, compare=False)

    def __post_init__(self) -> None:
        _post_init_multi_label(self)


@dataclass(eq=True, frozen=True)
class LabeledMultiSpan(Annotation):
    slices: Tuple[Tuple[int, int], ...]
    label: str
    score: float = field(default=1.0, compare=False)

    def __post_init__(self) -> None:
        _post_init_multi_span(self)
        _post_init_single_label(self)


@dataclass(eq=True, frozen=True)
class MultiLabeledMultiSpan(Annotation):
    slices: Tuple[Tuple[int, int], ...]
    label: Tuple[str, ...]
    score: Optional[Tuple[float, ...]] = field(default=None, compare=False)

    def __post_init__(self) -> None:
        _post_init_multi_span(self)
        _post_init_multi_label(self)


@dataclass(eq=True, frozen=True)
class BinaryRelation(Annotation):
    head: Annotation
    tail: Annotation
    label: str
    score: float = field(default=1.0, compare=False)

    def __post_init__(self) -> None:
        _post_init_single_label(self)


@dataclass(eq=True, frozen=True)
class MultiLabeledBinaryRelation(Annotation):
    head: Annotation
    tail: Annotation
    label: Tuple[str, ...]
    score: Optional[Tuple[float, ...]] = field(default=None, compare=False)

    def __post_init__(self) -> None:
        _post_init_multi_label(self)


@dataclass(eq=True, frozen=True)
class NaryRelation(Annotation):
    arguments: Tuple[Annotation, ...]
    roles: Tuple[str, ...]
    label: str
    score: float = field(default=1.0, compare=False)

    def __post_init__(self) -> None:
        _post_init_arguments_and_roles(self)
        _post_init_single_label(self)
