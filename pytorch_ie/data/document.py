from typing import Any, Dict, List, Optional, Tuple, Union


class Annotation:
    def __init__(
        self,
        label: Union[str, List[str]],
        score: Optional[Union[float, List[float]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._label = label
        self._metadata = metadata or {}

        is_multilabel = isinstance(label, list)

        if score is None:
            score = [1.0] * len(self.label) if is_multilabel else 1.0

        if is_multilabel:
            if not isinstance(score, list):
                raise ValueError("Multi-label requires score to be a list.")

            if len(label) != len(score):
                raise ValueError("Number of labels and scores must be equal.")
        else:
            if not isinstance(score, float):
                raise ValueError("Too many scores for label.")

        self._score = score
        self._is_multilabel = is_multilabel

    @property
    def label(self) -> Union[str, List[str]]:
        return self._label

    @property
    def label_single(self) -> str:
        assert isinstance(self._label, str), "this annotation has multiple labels"
        return self._label

    @property
    def labels(self) -> List[str]:
        if isinstance(self._label, list):
            return self._label
        elif isinstance(self._label, str):
            return [self._label]
        else:
            ValueError(
                f"the label has an unknown type: `{type(self._label)}`, it should be either `str` or "
                f"`list` (for multilabel setup)."
            )

    @property
    def score(self) -> Union[float, List[float]]:
        return self._score

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    @property
    def is_multilabel(self) -> bool:
        return self._is_multilabel

    @classmethod
    def from_dict(cls, dct: Dict[str, Any]) -> "Annotation":
        return cls(**dct)


# just for now as simple type shortcuts
AnnotationLayer = List[Annotation]
AnnotationCollection = Dict[str, List[Annotation]]


class Label(Annotation):
    def __init__(
        self,
        label: Union[str, List[str]],
        score: Optional[Union[float, List[float]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(label=label, score=score, metadata=metadata)

    def __repr__(self) -> str:
        return f"Label(label={self.label}, score={self.score})"


class LabeledMultiSpan(Annotation):
    def __init__(
        self,
        slices: List[Tuple[int, int]],
        label: Union[str, List[str]],
        score: Optional[Union[float, List[float]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(label=label, score=score, metadata=metadata)
        self.slices = slices

    def __repr__(self) -> str:
        return (
            f"LabeledMultiSpan(slices={self.slices}, label={self.label}, "
            f"score={self.score}, metadata={self.metadata})"
        )


class LabeledSpan(LabeledMultiSpan):
    def __init__(
        self,
        start: int,
        end: int,
        label: Union[str, List[str]],
        score: Optional[Union[float, List[float]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(slices=[(start, end)], label=label, score=score, metadata=metadata)

    @property
    def start(self):
        return self.slices[0][0]

    @property
    def end(self):
        return self.slices[-1][1]

    def __repr__(self) -> str:
        return (
            f"LabeledSpan(start={self.start}, end={self.end}, label={self.label}, "
            f"score={self.score}, metadata={self.metadata})"
        )


class BinaryRelation(Annotation):
    def __init__(
        self,
        head: LabeledSpan,
        tail: LabeledSpan,
        label: Union[str, List[str]],
        score: Optional[Union[float, List[float]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(label=label, score=score, metadata=metadata)
        self.head = head
        self.tail = tail

    def __repr__(self) -> str:
        return (
            f"BinaryRelation(head={self.head}, tail={self.tail}, label={self.label}, "
            f"score={self.score}, metadata={self.metadata})"
        )


class Document:
    def __init__(self, text: str, doc_id: Optional[str] = None) -> None:
        self._text = text
        self._id = doc_id
        self._metadata: Dict[str, Any] = {}
        self._annotations: AnnotationCollection = {}
        self._predictions: AnnotationCollection = {}

    @property
    def text(self) -> str:
        return self._text

    @property
    def id(self) -> Optional[str]:
        return self._id

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    def add_annotation(self, name: str, annotation: Annotation) -> None:
        if name not in self._annotations:
            self._annotations[name] = []

        self._annotations[name].append(annotation)

    def add_prediction(self, name: str, prediction: Annotation) -> None:
        if name not in self._predictions:
            self._predictions[name] = []

        self._predictions[name].append(prediction)

    # TODO: rework all getters for annotations (e.g. remove type: ignore)
    def annotations(self, name: str) -> Optional[AnnotationLayer]:
        return self._annotations.get(name, None)

    def span_annotations(self, name: str) -> Optional[List[LabeledSpan]]:
        return self.annotations(name=name)  # type: ignore

    def relation_annotations(self, name: str) -> Optional[List[BinaryRelation]]:
        return self.annotations(name=name)  # type: ignore

    def label_annotations(self, name: str) -> Optional[List[Label]]:
        return self.annotations(name=name)  # type: ignore

    def predictions(self, name: str) -> Optional[AnnotationLayer]:
        return self._predictions.get(name, None)

    def clear_predictions(self, name: str) -> None:
        if name in self._predictions:
            del self._predictions[name]

    def __repr__(self) -> str:
        return (
            f"Document(text={self.text}, annotations={self._annotations}, "
            f"predictions={self._predictions}, metadata={self.metadata})"
        )
