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
    cast,
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


# just for now as simple type shortcuts
# AnnotationLayer = List[Annotation]
# AnnotationCollection = Dict[str, List[Annotation]]


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


class AnnotationLayer(Generic[T_annotation]):
    def __init__(
        self,
        annotation_type: Optional[Type] = None,
        annotations: Optional[Collection[T_annotation]] = None,
    ):
        self._annotations: List[T_annotation] = []
        if annotation_type is not None:
            self._annotation_type = annotation_type
        else:
            if annotations is None or len(annotations) == 0:
                raise ValueError(
                    f"if no annotation type is provided, at least one annotation has to be given to "
                    f"infer the type for the new annotation layer"
                )
            self._annotation_type = type(list(annotations)[0])
        if annotations is not None:
            self.add(annotations)

    def add(self, annotation: Union[T_annotation, Collection[T_annotation]]):
        if isinstance(annotation, Annotation):
            annotations = [annotation]
        else:
            if not isinstance(annotation, Collection):
                raise TypeError(f"can only add a single Annotation or a collection of them")
            annotations = list(annotation)
        for ann in annotations:
            if not isinstance(ann, self._annotation_type):
                raise TypeError(
                    f"Annotation type mismatch. Expected: {self._annotation_type}, actual: {type(ann)}."
                )
            self._annotations.append(ann)

    def __iter__(self) -> Iterator[T_annotation]:
        return (a for a in self._annotations)

    def __len__(self) -> int:
        return len(self._annotations)

    def __getitem__(self, item) -> T_annotation:
        return self._annotations[item]

    def ensure_type(self, annotation_type: Type):
        if self._annotation_type != annotation_type:
            raise TypeError(
                f"Incorrect annotation type. Expected: {self._annotation_type.__name__}, "
                f"actual: {annotation_type.__name__}."
            )

    def __repr__(self) -> str:
        return f"AnnotationLayer(annotations={self._annotations}, annotation_type={self._annotation_type.__name__})"

    @property
    def as_spans(self) -> List[LabeledSpan]:
        self.ensure_type(LabeledSpan)
        return cast(List[LabeledSpan], self._annotations)

    @property
    def as_binary_relations(self) -> List[BinaryRelation]:
        self.ensure_type(BinaryRelation)
        return cast(List[BinaryRelation], self._annotations)

    @property
    def as_labels(self) -> List[Label]:
        self.ensure_type(Label)
        return cast(List[Label], self._annotations)


class AnnotationCollection:
    T_default = TypeVar("T_default")

    def __init__(self):
        self._layers: Dict[str, AnnotationLayer] = {}

    def add_layer(
        self,
        name: str,
        layer: Optional[AnnotationLayer] = None,
        annotations: Optional[List[Annotation]] = None,
        annotation_type: Optional[Type] = None,
        allow_overwrite: bool = False,
    ):
        if self.has_layer(name) and not allow_overwrite:
            raise ValueError(
                f"A layer with name {name} already exists. Use allow_overwrite=True to overwrite."
            )
        if layer is None:
            layer = AnnotationLayer(annotation_type=annotation_type, annotations=annotations)
        self._layers[name] = layer

    def has_layer(self, name: str) -> bool:
        return name in self._layers

    def add(self, name: str, annotation: Annotation, create_layer: bool = False):
        if not self.has_layer(name):
            if create_layer:
                self.add_layer(name, annotation_type=type(annotation))
            else:
                raise ValueError(f"layer with name {name} does not exist")
        self._layers[name].add(annotation)

    def get(self, name: str, default: T_default = None) -> Union[AnnotationLayer, T_default]:
        if self.has_layer(name):
            return self._layers[name]  # .cast()
        return default  # type: ignore

    def __getitem__(self, item) -> AnnotationLayer:
        return self._layers[item]  # .cast()

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


# not used
SpanLayer = AnnotationLayer[LabeledSpan]
BinaryRelationLayer = AnnotationLayer[BinaryRelation]
LabelLayer = AnnotationLayer[Label]
