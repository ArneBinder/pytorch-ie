from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, TypeVar, Union, overload

if TYPE_CHECKING:
    from pytorch_ie.document import Document


@dataclass(eq=True, frozen=True)
class Annotation:
    # _target: Optional[Union["Annotation", str]] = field(default=None, init=False, repr=False, hash=False, compare=False)
    _target: Optional[Union["Annotation", str]] = field(
        default=None, init=False, repr=False, hash=False
    )

    def set_target(self, value: Union["Annotation", str]):
        object.__setattr__(self, "_target", value)

    @property
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
        tmp_dct = dict(dct)
        tmp_dct.pop("id", None)
        return cls(**tmp_dct)


class SingleLabelMixin:
    def __post_init__(self) -> None:
        if not isinstance(self.label, str):
            raise ValueError("label must be a single string.")

        if not isinstance(self.score, float):
            raise ValueError("score must be a single float.")


class MultiLabelMixin:
    def __post_init__(self) -> None:
        if self.score is None:
            score = tuple([1.0] * len(self.label))
            object.__setattr__(self, "score", score)

        if isinstance(self.label, list):
            object.__setattr__(self, "label", tuple(self.label))

        if isinstance(self.score, list):
            object.__setattr__(self, "score", tuple(self.score))

        if len(self.label) != len(self.score):
            raise ValueError(
                f"Number of labels ({len(self.label)}) and scores ({len(self.score)}) must be equal."
            )


@dataclass(eq=True, frozen=True)
class Label(Annotation, SingleLabelMixin):
    label: str
    score: float = 1.0


@dataclass(eq=True, frozen=True)
class MultiLabel(Annotation, MultiLabelMixin):
    label: Tuple[str]
    score: Optional[Tuple[float]] = None


@dataclass(eq=True, frozen=True)
class Span(Annotation):
    start: int
    end: int

    @property
    def text(self) -> str:
        return self.target[self.start : self.end]


@dataclass(eq=True, frozen=True)
class LabeledSpan(Span, SingleLabelMixin):
    label: str
    score: float = 1.0


@dataclass(eq=True, frozen=True)
class MultiLabeledSpan(Span, MultiLabelMixin):
    label: Tuple[str]
    score: Optional[Tuple[float]] = None


@dataclass(eq=True, frozen=True)
class LabeledMultiSpan(Annotation, SingleLabelMixin):
    slices: Tuple[Tuple[int, int]]
    label: str
    score: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self.label, list):
            object.__setattr__(self, "slices", tuple(self.slices))


@dataclass(eq=True, frozen=True)
class MultiLabeledMultiSpan(Annotation, MultiLabelMixin):
    slices: Tuple[Tuple[int, int]]
    label: Tuple[str]
    score: Optional[Tuple[float]] = None


@dataclass(eq=True, frozen=True)
class BinaryRelation(Annotation, SingleLabelMixin):
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
        tmp_dct = dict(dct)
        tmp_dct.pop("id", None)

        head = tmp_dct["head"]
        tail = tmp_dct["tail"]

        if isinstance(head, int):
            if annotations is None:
                raise ValueError("Unable to resolve head reference without annotations.")

            tmp_dct["head"] = annotations[head][1]

        if isinstance(tail, int):
            if annotations is None:
                raise ValueError("Unable to resolve tail reference without annotations.")

            tmp_dct["tail"] = annotations[tail][1]

        return cls(**tmp_dct)


@dataclass(eq=True, frozen=True)
class MultiLabeledBinaryRelation(Annotation, MultiLabelMixin):
    head: Span
    tail: Span
    label: Tuple[str]
    score: Optional[Tuple[float]] = None

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
        tmp_dct = dict(dct)
        tmp_dct.pop("id", None)

        head = tmp_dct["head"]
        tail = tmp_dct["tail"]

        if isinstance(head, int):
            if annotations is None:
                raise ValueError("Unable to resolve head reference without annotations.")

            tmp_dct["head"] = annotations[head][1]

        if isinstance(tail, int):
            if annotations is None:
                raise ValueError("Unable to resolve tail reference without annotations.")

            tmp_dct["tail"] = annotations[tail][1]

        return cls(**tmp_dct)


T = TypeVar("T", covariant=True, bound=Annotation)


class PredictionList(Sequence[T]):
    def __init__(self, document: "Document", target: "str"):
        self._document = document
        self._target = target
        self._predictions: List[T] = []

    # TODO: check if the comparison logic is sufficient
    def __eq__(self, other: object) -> bool:
        return self._target == other._target and self._predictions == other._predictions

    @overload
    def __getitem__(self, idx: int) -> T:
        ...

    @overload
    def __getitem__(self, s: slice) -> Sequence[T]:
        ...

    def __getitem__(self, idx) -> T:
        return self._predictions[idx]

    def __len__(self) -> int:
        return len(self._predictions)

    def append(self, prediction: T) -> None:
        prediction.set_target(getattr(self._document, self._target))
        self._predictions.append(prediction)

    def __repr__(self) -> str:
        return f"PredictionList({str(self._predictions)})"

    def clear(self):
        for prediction in self._predictions:
            prediction.set_target(None)
        self._predictions = []


class AnnotationList(Sequence[T]):
    def __init__(self, document: "Document", target: "str"):
        self._document = document
        self._target = target
        self._annotations: List[T] = []
        self._predictions: PredictionList[T] = PredictionList(document, target)

    @property
    def predictions(self) -> PredictionList[T]:
        return self._predictions

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
        annotation.set_target(getattr(self._document, self._target))
        self._annotations.append(annotation)

    def __repr__(self) -> str:
        return f"AnnotationList({str(self._annotations)})"

    def clear(self):
        for annotation in self._annotations:
            annotation.set_target(None)
        self._annotations = []
