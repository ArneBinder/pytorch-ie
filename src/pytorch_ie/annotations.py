from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from pytorch_ie.core.document import Annotation, resolve_annotation


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


@dataclass(eq=True, frozen=True)
class Label(Annotation):
    label: str
    score: float = 1.0

    def __post_init__(self) -> None:
        _post_init_single_label(self)


@dataclass(eq=True, frozen=True)
class MultiLabel(Annotation):
    label: Tuple[str, ...]
    score: Optional[Tuple[float, ...]] = None

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
    score: float = 1.0

    def __post_init__(self) -> None:
        _post_init_single_label(self)


@dataclass(eq=True, frozen=True)
class MultiLabeledSpan(Span):
    label: Tuple[str, ...]
    score: Optional[Tuple[float, ...]] = None

    def __post_init__(self) -> None:
        _post_init_multi_label(self)


@dataclass(eq=True, frozen=True)
class LabeledMultiSpan(Annotation):
    slices: Tuple[Tuple[int, int], ...]
    label: str
    score: float = 1.0

    def __post_init__(self) -> None:
        _post_init_multi_span(self)
        _post_init_single_label(self)


@dataclass(eq=True, frozen=True)
class MultiLabeledMultiSpan(Annotation):
    slices: Tuple[Tuple[int, int], ...]
    label: Tuple[str, ...]
    score: Optional[Tuple[float, ...]] = None

    def __post_init__(self) -> None:
        _post_init_multi_span(self)
        _post_init_multi_label(self)


@dataclass(eq=True, frozen=True)
class BinaryRelation(Annotation):
    head: Span
    tail: Span
    label: str
    score: float = 1.0

    def __post_init__(self) -> None:
        _post_init_single_label(self)

    def asdict(self) -> Dict[str, Any]:
        dct = self._asdict(overrides={"head": self.head._id, "tail": self.tail._id})
        return dct

    @classmethod
    def fromdict(
        cls,
        dct: Dict[str, Any],
        annotation_store: Optional[Dict[int, Annotation]] = None,
    ):
        # copy to not modify the input
        tmp_dct = dict(dct)
        tmp_dct["head"] = resolve_annotation(tmp_dct["head"], store=annotation_store)
        tmp_dct["tail"] = resolve_annotation(tmp_dct["tail"], store=annotation_store)
        return super().fromdict(tmp_dct, annotation_store)


@dataclass(eq=True, frozen=True)
class MultiLabeledBinaryRelation(Annotation):
    head: Span
    tail: Span
    label: Tuple[str, ...]
    score: Optional[Tuple[float, ...]] = None

    def __post_init__(self) -> None:
        _post_init_multi_label(self)

    def asdict(self) -> Dict[str, Any]:
        # replace object references with object hashes
        dct = self._asdict(overrides={"head": self.head._id, "tail": self.tail._id})
        return dct

    @classmethod
    def fromdict(
        cls,
        dct: Dict[str, Any],
        annotation_store: Optional[Dict[int, "Annotation"]] = None,
    ):
        tmp_dct = dict(dct)
        tmp_dct.pop("_id", None)

        tmp_dct["head"] = resolve_annotation(tmp_dct["head"], store=annotation_store)
        tmp_dct["tail"] = resolve_annotation(tmp_dct["tail"], store=annotation_store)

        return cls(**tmp_dct)
