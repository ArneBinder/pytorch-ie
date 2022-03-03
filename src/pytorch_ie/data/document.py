from typing import (
    Any,
    Collection,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast, Iterable,
)


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
    This is a simple wrapper around List that provides some casting methods. It is totally optional
    to use them, they are just to ease typing. E.g. the following would not cause any trouble for mypy:

    layer = AnnotationLayer([SpanAnnotation(start=0, end=2, label="e1"), SpanAnnotation(start=4, end=5, label="e2")])
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

    @property
    def as_spans(self) -> List[LabeledSpan]:
        self._check_type(LabeledSpan)
        return cast(List[LabeledSpan], self)

    @property
    def as_binary_relations(self) -> List[BinaryRelation]:
        self._check_type(BinaryRelation)
        return cast(List[BinaryRelation], self)

    @property
    def as_labels(self) -> List[Label]:
        self._check_type(Label)
        return cast(List[Label], self)


T_layer_default = TypeVar("T_layer_default")


class AnnotationCollection:

    def __init__(self):
        self._layers: Dict[str, AnnotationLayer] = {}

    def add_layer(
        self,
        name: str,
        layer: Optional[AnnotationLayer] = None,
        annotations: Optional[List[Annotation]] = None,
        allow_overwrite: bool = False,
    ):
        if self.has_layer(name) and not allow_overwrite:
            raise ValueError(
                f"A layer with name {name} already exists. Use allow_overwrite=True to overwrite."
            )
        if layer is None:
            layer = AnnotationLayer(annotations or [])
        self._layers[name] = layer

    def has_layer(self, name: str) -> bool:
        return name in self._layers

    def add(self, name: str, annotation: Annotation, create_layer: bool = False):
        if not self.has_layer(name):
            if create_layer:
                self.add_layer(name)
            else:
                raise ValueError(f"layer with name {name} does not exist")
        self._layers[name].append(annotation)

    def get(self, name: str, default: T_layer_default) -> Union[AnnotationLayer, T_layer_default]:
        if self.has_layer(name):
            return self._layers[name]
        return default

    def __getitem__(self, item) -> AnnotationLayer:
        return self._layers[item]

    def __delitem__(self, key):
        del self._layers[key]

    def __repr__(self) -> str:
        return f"AnnotationCollection(layers={self._layers})"

    @property
    def named_layers(self) -> List[Tuple[str, AnnotationLayer]]:
        return list(self._layers.items())


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
        if self.predictions.has_layer(name):
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
    doc_id: Optional[str] = None,
    tokens: Optional[List[str]] = None,
    spans: Optional[Dict[str, List[LabeledSpan]]] = None,
    binary_relations: Optional[Dict[str, List[BinaryRelation]]] = None,
    assert_span_text: bool = False,
) -> Document:
    doc = Document(text=text, doc_id=doc_id)
    if tokens is not None:
        doc.metadata["tokens"] = tokens

    if spans is not None:
        for layer_name, layer_spans in spans.items():
            span_layer = AnnotationLayer[LabeledSpan](layer_spans)
            doc.annotations.add_layer(name=layer_name, layer=span_layer)
            if assert_span_text:
                for ann in doc.annotations[layer_name]:
                    _assert_span_text(doc, ann)
    if binary_relations is not None:
        for layer_name, layer_binary_relations in binary_relations.items():
            rel_layer = AnnotationLayer[BinaryRelation](layer_binary_relations)
            doc.annotations.add_layer(name=layer_name, layer=rel_layer)

    return doc


# This is currently not used.
# However, these aliases can be used to cast layers without the need for
# type specific getters (if we decide to strip them), e.g.:
#   entities = cast(SpanLayer, doc.annotations["entities"])
SpanLayer = AnnotationLayer[LabeledSpan]
BinaryRelationLayer = AnnotationLayer[BinaryRelation]
LabelLayer = AnnotationLayer[Label]
