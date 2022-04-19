from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, TypeVar, Union, overload

if TYPE_CHECKING:
    from pytorch_ie.data.document import Document


@dataclass(eq=True, frozen=True)
class Annotation:
    # _target: Optional[Union["Annotation", str]] = field(default=None, init=False, repr=False, hash=False, compare=False)
    _target: Optional[Union["Annotation", str]] = field(default=None, init=False, repr=False)

    def set_target(self, value: Union["Annotation", str]):
        object.__setattr__(self, "_target", value)

    def target(self) -> Union["Annotation", str]:
        return self._target

    def asdict(self) -> Dict[str, Any]:
        dct = asdict(self)
        dct["id"] = hash(self)
        del dct["_target"]
        return dct

    @classmethod
    def fromdict(
        cls,
        dct: Dict[str, Any],
        annotations: Optional[Dict[int, Tuple[str, "Annotation"]]] = None,
    ):
        return cls(**dct)


@dataclass(eq=True, frozen=True)
class Label(Annotation):
    label: str
    score: float = 1.0


@dataclass(eq=True, frozen=True)
class MultiLabel(Annotation):
    label: List[str] = field(default_factory=list)
    score: Optional[List[float]] = None

    def __post_init__(self) -> None:
        if self.score is None:
            self.score = [1.0] * len(self.label)

        if len(self.label) != len(self.score):
            raise ValueError(
                f"Number of labels ({len(self.label)}) and scores ({len(self.score)}) must be equal."
            )


@dataclass(eq=True, frozen=True)
class Span(Annotation):
    start: int
    end: int

    @property
    def text(self) -> str:
        return self._target[self.start : self.end]


@dataclass(eq=True, frozen=True)
class LabeledSpan(Span):
    label: str
    score: float = 1.0


@dataclass(eq=True, frozen=True)
class MultiLabeledSpan(Span):
    label: List[str] = field(default_factory=list)
    score: Optional[List[float]] = None

    def __post_init__(self) -> None:
        if self.score is None:
            self.score = [1.0] * len(self.label)

        if len(self.label) != len(self.score):
            raise ValueError(
                f"Number of labels ({len(self.label)}) and scores ({len(self.score)}) must be equal."
            )


dataclass(eq=True, frozen=True)
class LabeledMultiSpan(Annotation):
    slices: List[Tuple[int, int]] = field(default_factory=list)
    label: str
    score: float = 1.0

    def __post_init__(self) -> None:
        if self.score is None:
            self.score = [1.0] * len(self.label)

        if len(self.label) != len(self.score):
            raise ValueError(
                f"Number of labels ({len(self.label)}) and scores ({len(self.score)}) must be equal."
            )


dataclass(eq=True, frozen=True)
class MultiLabeledMultiSpan(Annotation):
    slices: List[Tuple[int, int]] = field(default_factory=list)
    label: List[str] = field(default_factory=list)
    score: Optional[List[float]] = None

    def __post_init__(self) -> None:
        if self.score is None:
            self.score = [1.0] * len(self.label)

        if len(self.label) != len(self.score):
            raise ValueError(
                f"Number of labels ({len(self.label)}) and scores ({len(self.score)}) must be equal."
            )


@dataclass(eq=True, frozen=True)
class BinaryRelation(Annotation):
    head: Span
    tail: Span
    label: str
    score: float = 1.0

    def asdict(self) -> Dict[str, Any]:
        dct = super().asdict()
        dct["head"] = hash(self.head)
        dct["tail"] = hash(self.tail)
        return dct

    @classmethod
    def fromdict(
        cls,
        dct: Dict[str, Any],
        annotations: Optional[Dict[int, Tuple[str, "Annotation"]]] = None,
    ):
        if annotations is not None:
            head_id = dct["head"]
            tail_id = dct["tail"]

            dct["head"] = annotations[head_id][1]
            dct["tail"] = annotations[tail_id][1]

        return cls(**dct)


@dataclass(eq=True, frozen=True)
class MultiLabeledBinaryRelation(Annotation):
    head: Span
    tail: Span
    label: List[str] = field(default_factory=list)
    score: Optional[List[float]] = None

    def __post_init__(self) -> None:
        if self.score is None:
            self.score = [1.0] * len(self.label)

        if len(self.label) != len(self.score):
            raise ValueError(
                f"Number of labels ({len(self.label)}) and scores ({len(self.score)}) must be equal."
            )

    def asdict(self) -> Dict[str, Any]:
        dct = super().asdict()

        # replace object references with object hashes
        dct["head"] = hash(self.head)
        dct["tail"] = hash(self.tail)
        return dct

    @classmethod
    def fromdict(
        cls,
        dct: Dict[str, Any],
        annotations: Optional[Dict[int, Tuple[str, "Annotation"]]] = None,
    ):
        if annotations is not None:
            head_id = dct["head"]
            tail_id = dct["tail"]

            # resolve annotation hashes to annotation objects
            dct["head"] = annotations[head_id][1]
            dct["tail"] = annotations[tail_id][1]

        return cls(**dct)


T = TypeVar("T", covariant=True, bound=Annotation)


class AnnotationList(Sequence[T]):
    def __init__(self, document: "Document", target: "str"):
        self._document = document
        self._target = target
        self._annotations: List[T] = []

    # TODO: check if the comparison logic is sufficient
    def __eq__(self, other: object) -> bool:
        return self._target == other._target and self._annotations == other._annotations

    @overload
    def __getitem__(self, idx: int) -> T:
        ...

    @overload
    def __getitem__(self, s: slice) -> Sequence[T]:
        ...

    def __getitem__(self, idx) -> T:
        return self._annotations[idx]

    def __len__(self) -> int:
        return len(self._annotations)

    def append(self, annotation: T) -> None:
        annotation.set_target(self._target)
        self._annotations.append(annotation)

    def __repr__(self) -> str:
        return f"AnnotationList({str(self._annotations)})"

    def clear(self):
        for annotation in self._annotations:
            annotation.set_target(None)
        self._annotations = []
