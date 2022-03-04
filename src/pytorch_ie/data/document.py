from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union, cast


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


T_annotation = TypeVar("T_annotation", bound=Annotation)


class AnnotationLayer(List[T_annotation]):
    """
    An AnnotationLayer is a special List with some sanity checks and typed getters. It is ensured that all
    entries have the same type.

    It is totally optional to use the typed getters, they are just available to
    ease typing, e.g. the following would not cause any trouble for mypy:

        layer = AnnotationLayer([SpanAnnotation(start=0, end=2, label="e1")])
        entity = layer.as_spans[0]
        start, end = entity.start, entity.end  # access to .end and .start would cause issues otherwise
    """

    def _check_type(self, type_to_check: Type):
        if len(self) > 0 and not isinstance(self[0], type_to_check):
            raise TypeError(
                f"Entry caused a type mismatch. Expected type: {type(self[0])}, actual type: {type_to_check}."
            )

    def append(self, __object: T_annotation) -> None:
        self._check_type(type(__object))
        super().append(__object)

    def extend(self, __iterable: Iterable[T_annotation]) -> None:
        for e in __iterable:
            self.append(e)

    def __setitem__(self, key, value):
        self._check_type(type(value))
        super().__setitem__(i=key, o=value)

    def as_type(self, annotation_type: Type) -> List[T_annotation]:
        self._check_type(annotation_type)
        return self

    @property
    def as_spans(self) -> List[LabeledSpan]:
        return cast(List[LabeledSpan], self.as_type(LabeledSpan))

    @property
    def as_binary_relations(self) -> List[BinaryRelation]:
        return cast(List[BinaryRelation], self.as_type(BinaryRelation))

    @property
    def as_labels(self) -> List[Label]:
        return cast(List[Label], self.as_type(Label))


class AnnotationCollection(Dict[str, AnnotationLayer]):
    """
    An `AnnotationCollection` holds a mapping from layer names to `AnnotationLayers`. However, it
    also provides an `add` method to directly add an Annotation to a certain layer and create that if necessary.
    """

    def has_layer(self, name: str) -> bool:
        return name in self

    def add(self, name: str, annotation: Optional[Annotation] = None, create_layer: bool = False):
        if name not in self:
            if create_layer:
                self[name] = AnnotationLayer()
            else:
                raise ValueError(f"layer with name {name} does not exist")
        if annotation is not None:
            self[name].append(annotation)

    @property
    def named_layers(self) -> List[Tuple[str, AnnotationLayer]]:
        return list(self.items())


class Document:
    def __init__(self, text: str, doc_id: Optional[str] = None) -> None:
        self._text = text
        self._id = doc_id
        self._metadata: Dict[str, Any] = {}
        self.annotations = AnnotationCollection()
        self.predictions = AnnotationCollection()

    @property
    def text(self) -> str:
        return self._text

    @property
    def id(self) -> Optional[str]:
        return self._id

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    def add_annotation(self, name: str, annotation: Annotation):
        self.annotations.add(name=name, annotation=annotation, create_layer=True)

    def add_prediction(self, name: str, prediction: Annotation):
        self.predictions.add(name=name, annotation=prediction, create_layer=True)

    def clear_predictions(self, name: str) -> None:
        if name in self.predictions:
            del self.predictions[name]

    def __repr__(self) -> str:
        return (
            f"Document(text={self.text}, annotations={self.annotations}, "
            f"predictions={self.predictions}, metadata={self.metadata})"
        )


def _assert_span_text(doc: Document, span: LabeledSpan):
    assert doc.text[span.start : span.end] == span.metadata["text"]


def construct_document(
    text: str,
    spans: Optional[Dict[str, Iterable[LabeledSpan]]] = None,
    binary_relations: Optional[Dict[str, Iterable[BinaryRelation]]] = None,
    doc_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    assert_span_text: bool = False,
) -> Document:
    """
    Construct a `Document` from at least a text. If provided, add span and binary relation annotations,
    a document id and metadata.

    Args:
        text: the document text
        spans: a mapping from layer names to span annotations
        binary_relations: a mapping from layer names to binary relation annotations
        doc_id: a document id
        metadata: the content of this dictionary is added to the document metadata
        assert_span_text: If this is True, each span annotation in spans has to have an entry "text" in
            its metadata that contains the respective text slice from the document text. This is useful
            when creating spans with expected content (e.g. when writing tests).

    returns:
        The constructed document.
    """
    doc = Document(text=text, doc_id=doc_id)
    if metadata is not None:
        doc.metadata.update(metadata)

    if spans is not None:
        for layer_name, layer_spans in spans.items():
            doc.annotations[layer_name] = AnnotationLayer[LabeledSpan](layer_spans)
            if assert_span_text:
                for ann in doc.annotations[layer_name]:
                    _assert_span_text(doc, ann)
    if binary_relations is not None:
        for layer_name, layer_binary_relations in binary_relations.items():
            doc.annotations[layer_name] = AnnotationLayer[BinaryRelation](layer_binary_relations)

    return doc


# This is currently not used.
# However, these aliases can be used to cast layers without the need for
# type specific getters (if we decide to strip them), e.g.:
#   entities = cast(SpanLayer, doc.annotations["entities"])
SpanLayer = AnnotationLayer[LabeledSpan]
BinaryRelationLayer = AnnotationLayer[BinaryRelation]
LabelLayer = AnnotationLayer[Label]
