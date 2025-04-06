from dataclasses import dataclass, field
from typing import Any, Tuple

from pie_core.document import Annotation


@dataclass(frozen=True)
class WithPostInit:
    """Base class for annotations that require post-initialization checks."""

    def __post_init__(self) -> None:
        pass


@dataclass(frozen=True)
class WithLabel(WithPostInit):
    label: str
    score: float = field(default=1.0, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.label, str):
            raise ValueError("label must be a single string.")

        if not isinstance(self.score, float):
            raise ValueError("score must be a single float.")

        super().__post_init__()


@dataclass(frozen=True)
class WithMultiLabel(WithPostInit):
    label: Tuple[str, ...]
    score: Tuple[float, ...] = field(default=(), compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.label, tuple):
            object.__setattr__(self, "label", tuple(self.label))

        if len(self.score) == 0:
            if len(self.label) == 0:
                raise ValueError("label and score cannot be empty.")
            score = tuple([1.0] * len(self.label))
            object.__setattr__(self, "score", score)
        else:
            if not isinstance(self.score, tuple):
                object.__setattr__(self, "score", tuple(self.score))

            if len(self.label) != len(self.score):
                raise ValueError(
                    f"Number of labels ({len(self.label)}) and scores "
                    f"({len(self.score)}) must be equal."
                )

        super().__post_init__()


@dataclass(frozen=True)
class Label(WithLabel, Annotation):

    def resolve(self) -> Any:
        return self.label


@dataclass(frozen=True)
class MultiLabel(WithMultiLabel, Annotation):

    def resolve(self) -> Any:
        return self.label


@dataclass(frozen=True)
class Span(Annotation):
    start: int
    end: int

    def __str__(self) -> str:
        if not self.is_attached:
            return super().__str__()
        return str(self.target[self.start : self.end])

    def resolve(self) -> Any:
        if self.is_attached:
            return self.target[self.start : self.end]
        else:
            raise ValueError(f"{self} is not attached to a target.")


@dataclass(frozen=True)
class LabeledSpan(WithLabel, Span):

    def resolve(self) -> Any:
        return self.label, super().resolve()


@dataclass(frozen=True)
class MultiLabeledSpan(WithMultiLabel, Span):

    def resolve(self) -> Any:
        return self.label, super().resolve()


@dataclass(frozen=True)
class MultiSpan(WithPostInit, Annotation):
    slices: Tuple[Tuple[int, int], ...]

    def __post_init__(self) -> None:
        if isinstance(self.slices, list):
            object.__setattr__(self, "slices", tuple(tuple(s) for s in self.slices))

        super().__post_init__()

    def __str__(self) -> str:
        if not self.is_attached:
            return super().__str__()
        else:
            return str(self.resolve())

    def resolve(self) -> Any:
        if self.is_attached:
            return tuple(self.target[start:end] for start, end in self.slices)
        else:
            raise ValueError(f"{self} is not attached to a target.")


@dataclass(frozen=True)
class LabeledMultiSpan(WithLabel, MultiSpan):

    def resolve(self) -> Any:
        return self.label, super().resolve()


@dataclass(frozen=True)
class AnnotationWithHeadAndTail(Annotation):
    head: Annotation
    tail: Annotation

    def resolve(self) -> Any:
        return self.head.resolve(), self.tail.resolve()


@dataclass(frozen=True)
class BinaryRelation(WithLabel, AnnotationWithHeadAndTail):

    def resolve(self) -> Any:
        return self.label, super().resolve()


@dataclass(frozen=True)
class MultiLabeledBinaryRelation(WithMultiLabel, AnnotationWithHeadAndTail):

    def resolve(self) -> Any:
        return self.label, super().resolve()


@dataclass(frozen=True)
class AnnotationWithArgumentsAndRoles(WithPostInit, Annotation):
    arguments: Tuple[Annotation, ...]
    roles: Tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.arguments) != len(self.roles):
            raise ValueError(
                f"Number of arguments ({len(self.arguments)}) and roles "
                f"({len(self.roles)}) must be equal"
            )
        if not isinstance(self.arguments, tuple):
            object.__setattr__(self, "arguments", tuple(self.arguments))
        if not isinstance(self.roles, tuple):
            object.__setattr__(self, "roles", tuple(self.roles))

        super().__post_init__()

    def resolve(self) -> Any:
        return tuple((role, arg.resolve()) for role, arg in zip(self.roles, self.arguments))


@dataclass(frozen=True)
class NaryRelation(WithLabel, AnnotationWithArgumentsAndRoles):

    def resolve(self) -> Any:
        return self.label, super().resolve()
