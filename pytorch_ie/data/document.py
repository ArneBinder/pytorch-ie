from typing import Any, Dict, List, Optional, Union


class Annotation:
    def __init__(
        self,
        label: Union[str, List[str]],
        score: Optional[Union[float, List[float]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._label = label

        if self.is_multilabel:
            if score is not None and len(label) != len(score):
                raise ValueError("Number of labels and scores must be equal.")
        else:
            if score is not None and not isinstance(score, float):
                raise ValueError("To many scores for label.")

        self._score = score
        self._metadata = metadata or {}

    @property
    def label(self) -> Union[str, List[str]]:
        return self._label

    @property
    def score(self) -> Union[str, List[str]]:
        if self._score is None:
            return [1.0 for _ in self._label] if self.is_multilabel else 1.0
        return self._score

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    @property
    def is_multilabel(self) -> bool:
        return not isinstance(self._label, str)


class Label(Annotation):
    def __init__(
        self,
        label: Union[str, List[str]],
        score: Optional[Union[float, List[float]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(label=label, score=score, metadata=metadata)

    def __repr__(self) -> str:
        return f"Label(label={self.label}, score={self.score:.2f})"


class LabeledSpan(Annotation):
    def __init__(
        self,
        start: int,
        end: int,
        label: Union[str, List[str]],
        score: Optional[Union[float, List[float]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(label=label, score=score, metadata=metadata)
        self.start = start
        self.end = end

    def __repr__(self) -> str:
        return f"LabeledSpan(start={self.start}, end={self.end}, label={self.label}, metadata={self.metadata})"

    @classmethod
    def from_dict(cls, dct: Dict[str, Any]) -> "LabeledSpan":
        return cls(**dct)


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
        return f"BinaryRelation(head={self.head}, tail={self.tail}, label={self.label}, metadata={self.metadata})"

    @classmethod
    def from_dict(cls, dct: Dict[str, Any]) -> "LabeledSpan":
        return cls(**dct)


class Document:
    def __init__(self, text: str, doc_id: Optional[str] = None) -> None:
        self._text = text
        self._id = doc_id
        self._metadata = {}
        self._annotations: Dict[str, List[Annotation]] = {}
        self._predictions: Dict[str, List[Annotation]] = {}

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

    def annotations(self, name: str) -> Optional[List[Annotation]]:
        return self._annotations.get(name, [])

    def predictions(self, name: str) -> Optional[List[Annotation]]:
        return self._predictions.get(name, [])

    def clear_predictions(self, name: str) -> None:
        if name in self._predictions:
            del self._predictions[name]

    def __repr__(self) -> str:
        return f"Document(text={self.text}, annotations={self._annotations}, predictions={self._predictions}, metadata={self.metadata})"


# just for now as types
AnnotationLayer = List[Annotation]
AnnotationCollection = Dict[str, List[Annotation]]
