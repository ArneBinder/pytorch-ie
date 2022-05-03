from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from pytorch_ie.core.document import Annotation


def _validate_single_label(self):
    if not isinstance(self.label, str):
        raise ValueError("label must be a single string.")

    if not isinstance(self.score, float):
        raise ValueError("score must be a single float.")


def _validate_multi_label(self):
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


@dataclass(eq=True, frozen=True)
class Label(Annotation):
    label: str
    score: float = 1.0

    def __post_init__(self) -> None:
        _validate_single_label(self)


@dataclass(eq=True, frozen=True)
class MultiLabel(Annotation):
    label: Tuple[str]
    score: Optional[Tuple[float]] = None

    def __post_init__(self) -> None:
        _validate_multi_label(self)


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
    score: float = 1.0

    def __post_init__(self) -> None:
        _validate_single_label(self)


@dataclass(eq=True, frozen=True)
class MultiLabeledSpan(Span):
    label: Tuple[str]
    score: Optional[Tuple[float]] = None

    def __post_init__(self) -> None:
        _validate_multi_label(self)


@dataclass(eq=True, frozen=True)
class LabeledMultiSpan(Annotation):
    slices: Tuple[Tuple[int, int]]
    label: str
    score: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.label, list):
            object.__setattr__(self, "slices", tuple(self.slices))

        _validate_single_label(self)


@dataclass(eq=True, frozen=True)
class MultiLabeledMultiSpan(Annotation):
    slices: Tuple[Tuple[int, int]]
    label: Tuple[str]
    score: Optional[Tuple[float]] = None

    def __post_init__(self) -> None:
        if isinstance(self.label, list):
            object.__setattr__(self, "slices", tuple(self.slices))

        _validate_multi_label(self)


@dataclass(eq=True, frozen=True)
class BinaryRelation(Annotation):
    head: Span
    tail: Span
    label: str
    score: float = 1.0

    def __post_init__(self) -> None:
        _validate_single_label(self)

    def asdict(self) -> Dict[str, Any]:
        dct = super().asdict()
        dct["head"] = hash(self.head)
        dct["tail"] = hash(self.tail)
        return dct

    @classmethod
    def fromdict(
        cls,
        dct: Dict[str, Any],
        annotation_store: Optional[Dict[int, Tuple[str, "Annotation"]]] = None,
    ):
        tmp_dct = dict(dct)
        tmp_dct.pop("_id", None)

        head = tmp_dct["head"]
        tail = tmp_dct["tail"]

        if isinstance(head, int):
            if annotation_store is None:
                raise ValueError("Unable to resolve head reference without annotation_store.")

            tmp_dct["head"] = annotation_store[head][1]

        if isinstance(tail, int):
            if annotation_store is None:
                raise ValueError("Unable to resolve tail reference without annotation_store.")

            tmp_dct["tail"] = annotation_store[tail][1]

        return cls(**tmp_dct)


@dataclass(eq=True, frozen=True)
class MultiLabeledBinaryRelation(Annotation):
    head: Span
    tail: Span
    label: Tuple[str]
    score: Optional[Tuple[float]] = None

    def __post_init__(self) -> None:
        _validate_multi_label(self)

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
        annotation_store: Optional[Dict[int, Tuple[str, "Annotation"]]] = None,
    ):
        tmp_dct = dict(dct)
        tmp_dct.pop("_id", None)

        head = tmp_dct["head"]
        tail = tmp_dct["tail"]

        if isinstance(head, int):
            if annotation_store is None:
                raise ValueError("Unable to resolve head reference without annotation_store.")

            tmp_dct["head"] = annotation_store[head][1]

        if isinstance(tail, int):
            if annotation_store is None:
                raise ValueError("Unable to resolve tail reference without annotation_store.")

            tmp_dct["tail"] = annotation_store[tail][1]

        return cls(**tmp_dct)
