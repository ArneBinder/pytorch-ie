from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
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


# simple list for now
AnnotationLayer = list
T_annotation = TypeVar("T_annotation", bound=Annotation)


class TypedAnnotationCollection(Generic[T_annotation], Dict[str, AnnotationLayer[T_annotation]]):
    """
    An `AnnotationCollection` holds a mapping from layer names to `AnnotationLayers`. However, it
    also provides an `add` method to directly add an Annotation to a certain layer and create that if necessary.
    """

    def has_layer(self, name: str) -> bool:
        return name in self

    def add(self, name: str, annotation: T_annotation):
        if not self.has_layer(name=name):
            self.create_layer(name=name)
        self[name].append(annotation)

    def create_layer(self, name: str, allow_exists: bool = False) -> AnnotationLayer[T_annotation]:
        if self.has_layer(name) and not allow_exists:
            raise ValueError(f"layer with name {name} already exists")
        self[name] = AnnotationLayer[T_annotation]()
        return self[name]

    @property
    def named_layers(self) -> Sequence[Tuple[str, AnnotationLayer[T_annotation]]]:
        return [item for item in self.items()]


class AnnotationCollection:
    def __init__(self):
        self.labels = TypedAnnotationCollection[Label]()
        self.spans = TypedAnnotationCollection[LabeledSpan]()
        self.binary_relations = TypedAnnotationCollection[BinaryRelation]()

        self._types_to_collections = {
            Label: self.labels,
            LabeledSpan: self.spans,
            BinaryRelation: self.binary_relations,
        }

    def add(self, name: str, annotation: Annotation):
        collection = self._types_to_collections.get(type(annotation))
        if collection is None:
            raise TypeError(f"annotation has unknown type: {type(annotation)}")
        collection.add(name=name, annotation=annotation)

    @property
    def typed_collections(self) -> Sequence[Tuple[Type, TypedAnnotationCollection]]:
        return [item for item in self._types_to_collections.items()]

    @property
    def typed_named_layers(self) -> Sequence[Tuple[Type, str, AnnotationLayer]]:
        res = []
        for base_type, typed_collection in self.typed_collections:
            res.extend(
                [(base_type, name, layer) for (name, layer) in typed_collection.named_layers]
            )
        return res

    def __repr__(self) -> str:
        return (
            f"Document(labels={self.labels}, spans={self.spans}, "
            f"binary_relations={self.binary_relations})"
        )


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
        self.annotations.add(name=name, annotation=annotation)

    def add_prediction(self, name: str, prediction: Annotation):
        self.predictions.add(name=name, annotation=prediction)

    def clear_predictions(self, name: str) -> None:
        # TODO: should we respect the base_type?
        for base_type, collection in self.predictions.typed_collections:
            if name in collection:
                del collection[name]

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
            doc.annotations.spans[layer_name] = AnnotationLayer[LabeledSpan](layer_spans)
            if assert_span_text:
                for ann in doc.annotations.spans[layer_name]:
                    _assert_span_text(doc, ann)
    if binary_relations is not None:
        for layer_name, layer_binary_relations in binary_relations.items():
            doc.annotations.binary_relations[layer_name] = AnnotationLayer[BinaryRelation](
                layer_binary_relations
            )

    return doc


# This is currently not used.
# However, these aliases can be used to cast layers without the need for
# type specific getters (if we decide to strip them), e.g.:
#   entities = cast(SpanLayer, doc.annotations["entities"])
SpanLayer = AnnotationLayer[LabeledSpan]
BinaryRelationLayer = AnnotationLayer[BinaryRelation]
LabelLayer = AnnotationLayer[Label]
