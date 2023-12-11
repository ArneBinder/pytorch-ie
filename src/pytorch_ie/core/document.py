import dataclasses
import logging
import typing
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import _asdict_inner  # type: ignore
from typing import (
    Any,
    ClassVar,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
    overload,
)

logger = logging.getLogger(__name__)


def _enumerate_dependencies(
    resolved: List[str],
    dependency_graph: Dict[str, List[str]],
    nodes: List[str],
    current_path: Optional[Set[str]] = None,
):
    current_path = current_path or set()
    for node in nodes:
        if node in current_path:
            raise ValueError(f"circular dependency detected at node: {node}")
        if node not in resolved:
            # terminal nodes
            if node not in dependency_graph:
                resolved.append(node)
            # nodes with dependencies
            else:
                # enumerate all dependencies first, then append itself
                _enumerate_dependencies(
                    resolved,
                    dependency_graph,
                    nodes=dependency_graph[node],
                    current_path=current_path | {node},
                )
                resolved.append(node)


def _is_optional_type(t: typing.Type) -> bool:
    type_origin = typing.get_origin(t)
    type_args = typing.get_args(t)
    return type_origin is typing.Union and len(type_args) == 2 and type(None) in type_args


def _is_optional_annotation_type(t: typing.Type) -> bool:
    is_optional = _is_optional_type(t)
    if not is_optional:
        return False
    return _is_annotation_type(typing.get_args(t)[0])


def _is_annotation_type(t: Any) -> bool:
    return type(t) == type and issubclass(t, Annotation)


def _contains_annotation_type(t: Any) -> bool:
    if _is_annotation_type(t):
        return True
    type_args = typing.get_args(t)
    return any(_contains_annotation_type(type_arg) for type_arg in type_args)


def _is_tuple_of_annotation_types(t: Any) -> bool:
    type_args = typing.get_args(t)
    if typing.get_origin(t) == tuple and _is_annotation_type(type_args[0]):
        if not (
            type_args[1] == Ellipsis
            or all(issubclass(type_arg, Annotation) for type_arg in type_args)
        ):
            raise TypeError(
                f"only tuples that do not mix Annotations with other types are supported"
            )
        return True
    else:
        return False


def _get_reference_fields_and_container_types(
    annotation_class: typing.Type["Annotation"],
) -> Dict[str, Any]:
    containers: Dict[str, Any] = {}
    for field in dataclasses.fields(annotation_class):
        if field.name == "_targets":
            continue
        if isinstance(field.type, type):
            field_type = field.type
        else:
            field_type = typing.get_type_hints(annotation_class)[field.name]
        if not _contains_annotation_type(field_type):
            continue
        if _is_optional_annotation_type(field_type):
            containers[field.name] = typing.Optional
            continue
        if _is_annotation_type(field_type):
            containers[field.name] = None
            continue
        if _is_tuple_of_annotation_types(field_type):
            containers[field.name] = tuple
            continue
        annot_name = annotation_class.__name__
        raise TypeError(
            f"The type '{field_type}' of the field '{field.name}' from Annotation subclass '{annot_name}' can not "
            f"be handled automatically. For automatic handling, type constructs that contain any Annotation subclasses "
            f"need to be either (1) pure subclasses of Annotation, (2) tuples of Annotation subclasses, or their "
            f"optional variants (examples: 1) Span, 2) Tuple[Span, ...], 3) Optional[Span]). Is the defined type "
            f"really the one you want to use? If so, consider to overwrite "
            f"{annot_name}.asdict() and {annot_name}.fromdict() by your own."
        )

    return containers


def _get_annotation_fields(fields: List[dataclasses.Field]) -> Set[dataclasses.Field]:
    # this was broken, so we raise an exception for now
    # return {f for f in fields if typing.get_origin(f.type) is AnnotationLayer}
    raise Exception(
        "_get_annotation_fields() is broken, please use Document.annotation_fields() instead"
    )


def annotation_field(
    target: Optional[str] = None,
    targets: Optional[List[str]] = None,
    named_targets: Optional[Dict[str, str]] = None,
):
    """
    We allow 3 variants to pass targets:
    1) as single value `target: str`: this works if only one target is required,
    2) as list of target names `targets: List[str]`: this works if multiple targets are required, but the
        respective Annotation class does _not_ define `TARGET_NAMES` (this disallows the usage of named_targets
        in the Annotation class), and
    3) as a mapping `named_targets: Dict[str, str]` from entries in Annotation.TARGET_NAMES to field names of the
        Document: This should be used if the respective Annotation class defines `TARGET_NAMES`, and thus,
        makes use of the `named_targets` property.
    """
    target_names = None
    new_targets = []
    if target is not None:
        if targets is not None or named_targets is not None:
            raise ValueError(f"only one of target, targets or named_targets can be set")
        new_targets = [target]
    if targets is not None:
        if target is not None or named_targets is not None:
            raise ValueError(f"only one of target, targets or named_targets can be set")
        new_targets = targets
    if named_targets is not None:
        if target is not None or targets is not None:
            raise ValueError(f"only one of target, targets or named_targets can be set")
        new_targets = list(named_targets.values())
        target_names = list(named_targets.keys())
    return dataclasses.field(
        metadata=dict(targets=new_targets, target_names=target_names), init=False, repr=False
    )


# for now, we only have annotation lists and texts
TARGET_TYPE = Union["AnnotationLayer", str]


@dataclasses.dataclass(eq=True, frozen=True)
class Annotation:
    _targets: Optional[Tuple[TARGET_TYPE, ...]] = dataclasses.field(
        default=None, init=False, repr=False, hash=False
    )
    TARGET_NAMES: ClassVar[Optional[Tuple[str, ...]]] = None

    def set_targets(self, value: Optional[Tuple[TARGET_TYPE, ...]]):
        if value is not None and self._targets is not None:
            raise ValueError(
                f"Annotation already has assigned targets. Clear the "
                f"annotation list container or remove the annotation with pop() "
                f"to assign it to a new annotation list with other targets."
            )
        object.__setattr__(self, "_targets", value)

    @property
    def non_comparison_fields_and_values(self) -> Tuple[Tuple[str, Any], ...]:
        return tuple(
            (f.name, getattr(self, f.name)) for f in dataclasses.fields(self) if not f.compare
        )

    @property
    def _id(self) -> int:
        # also calculate the hash over the non-comparison fields for id creation because otherwise
        # these fields would not be considered (per default, hash=false for non-comparison fields)
        # and (de-)serialization would de-duplicate annotations that vary just in these fields
        return hash((self, self.non_comparison_fields_and_values))

    @property
    def target(self) -> Optional[TARGET_TYPE]:
        if self._targets is None:
            return None
        if len(self._targets) == 0:
            raise ValueError(f"annotation has no target")
        if len(self._targets) > 1:
            raise ValueError(
                f"annotation has multiple targets, target is not defined in this case"
            )
        return self._targets[0]

    @property
    def targets(self) -> Optional[Tuple[TARGET_TYPE, ...]]:
        return self._targets

    @property
    def named_targets(self) -> Dict[str, TARGET_TYPE]:
        if self._targets is None:
            raise ValueError(
                f"targets is not set (this annotation may be not yet attached to a document)"
            )
        if self.TARGET_NAMES is None:
            raise TypeError(f"no TARGET_NAMES defined")
        return {name: self._targets[i] for i, name in enumerate(self.TARGET_NAMES)}

    def _asdict(
        self,
        exclude_fields: Optional[List[str]] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        result: List[Tuple[str, Any]] = []
        _exclude_fields = set(exclude_fields) if exclude_fields is not None else set()
        _exclude_fields.add("_targets")
        if overrides is not None:
            _exclude_fields.update(overrides)
            result.extend(overrides.items())
        for f in dataclasses.fields(self):
            if f.name in _exclude_fields:
                continue
            field_value = getattr(self, f.name)
            value = _asdict_inner(field_value, dict)
            result.append((f.name, value))
        dct = dict(result)
        dct["_id"] = self._id
        return dct

    def asdict(self) -> Dict[str, Any]:
        overrides = {}
        reference_fields_with_container_type = _get_reference_fields_and_container_types(
            type(self)
        )
        for field_name, container_type in reference_fields_with_container_type.items():
            if container_type is None:
                overrides[field_name] = getattr(self, field_name)._id
            elif container_type == typing.Optional:
                field_value = getattr(self, field_name)
                overrides[field_name] = None if field_value is None else field_value._id
            elif container_type == tuple:
                # save as list to be json compatible
                overrides[field_name] = [anno._id for anno in getattr(self, field_name)]
            else:
                raise Exception(f"unknown annotation container type: {container_type}")

        dct = self._asdict(overrides=overrides)
        return dct

    @classmethod
    def fromdict(
        cls,
        dct: Dict[str, Any],
        annotation_store: Optional[Dict[int, "Annotation"]] = None,
    ):
        tmp_dct = dict(dct)
        reference_fields_with_container_type = _get_reference_fields_and_container_types(cls)
        for field_name, container_type in reference_fields_with_container_type.items():
            if container_type is None:
                tmp_dct[field_name] = resolve_annotation(
                    tmp_dct[field_name], store=annotation_store
                )
            elif container_type == typing.Optional:
                if tmp_dct[field_name] is None:
                    tmp_dct[field_name] = None
                else:
                    tmp_dct[field_name] = resolve_annotation(
                        tmp_dct[field_name], store=annotation_store
                    )
            elif container_type == tuple:
                tmp_dct[field_name] = tuple(
                    resolve_annotation(anno_dct, store=annotation_store)
                    for anno_dct in tmp_dct[field_name]
                )
            else:
                raise Exception(f"unknown annotation container type: {container_type}")

        tmp_dct.pop("_id", None)
        return cls(**tmp_dct)

    @property
    def is_attached(self) -> bool:
        return self._targets is not None

    def copy(self, **overrides) -> "Annotation":
        """
        Create a detached copy of the annotation with the same values as the original.

        :param overrides: keyword arguments to override the values of the original annotation
        :return: a detached copy of the annotation
        """
        kwargs = {}
        for f in dataclasses.fields(self):
            if f.name == "_targets":
                continue
            kwargs[f.name] = getattr(self, f.name)
        kwargs.update(overrides)
        return type(self)(**kwargs)

    def copy_with_store(
        self, override_annotation_store: Dict[int, "Annotation"], invalid_annotation_ids: Set[int]
    ) -> Optional["Annotation"]:
        """
        Create a detached copy of the annotation, but replace references to other annotations
        with entries from an override annotation store.
        Note: For now, fields are only allowed to directly reference an annotation or a tuple of annotations,
         but not any nested data structure that contains annotations.

        Args:
            :param override_annotation_store: the annotation store to use, a mapping from original *annotation ids*
                to the new *annotations*
            :param invalid_annotation_ids: a set of annotation ids that are not valid anymore and so any Annotation that
                references one of these ids will be discarded
            :return: a detached copy of the annotation or None if the annotation contains references to invalid
                annotations
        """
        overrides: Dict[str, Any] = {}
        for f in dataclasses.fields(self):
            if f.name == "_targets":
                continue
            field_value = getattr(self, f.name)
            if isinstance(field_value, Annotation):
                if field_value._id in invalid_annotation_ids:
                    return None
                overrides[f.name] = override_annotation_store.get(field_value._id, field_value)
            elif isinstance(field_value, tuple):
                if any(
                    maybe_anno._id in invalid_annotation_ids
                    for maybe_anno in field_value
                    if isinstance(maybe_anno, Annotation)
                ):
                    return None
                overrides[f.name] = tuple(
                    override_annotation_store.get(maybe_anno._id, maybe_anno)
                    if isinstance(maybe_anno, Annotation)
                    else maybe_anno
                    for maybe_anno in field_value
                )
            elif isinstance(field_value, (int, float, str, bool, type(None))):
                continue
            else:
                raise Exception(f"unknown annotation field type: {type(field_value)}")

        return self.copy(**overrides)


T = TypeVar("T", covariant=False, bound="Annotation")


class BaseAnnotationList(Sequence[T]):
    def __init__(self, document: "Document", targets: List[str]):
        self._document = document
        self._targets = targets
        self._annotations: List[T] = []

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseAnnotationList):
            return NotImplemented

        return self._targets == other._targets and self._annotations == other._annotations

    @overload
    def __getitem__(self, index: int) -> T:
        ...

    @overload
    def __getitem__(self, s: slice) -> List[T]:
        ...

    def __getitem__(self, index: Union[int, slice]) -> Union[T, List[T]]:
        return self._annotations[index]

    def __len__(self) -> int:
        return len(self._annotations)

    def append(self, annotation: T) -> None:
        targets = tuple(getattr(self._document, target_name) for target_name in self._targets)
        annotation.set_targets(targets)
        self._annotations.append(annotation)

    def extend(self, annotations: Iterable[T]) -> None:
        for annotation in annotations:
            self.append(annotation)

    def __repr__(self) -> str:
        return f"BaseAnnotationList({str(self._annotations)})"

    def clear(self) -> List[T]:
        """
        Detach all annotations from the layer and return them.
        """
        result = list(self._annotations)
        for annotation in self._annotations:
            annotation.set_targets(None)
        self._annotations = []
        return result

    def pop(self, index: int = -1) -> T:
        ann = self._annotations.pop(index)
        ann.set_targets(None)
        return ann

    @property
    def targets(self) -> dict[str, Any]:
        return {
            target_field_name: getattr(self._document, target_field_name)
            for target_field_name in self._targets
        }

    @property
    def target_names(self) -> List[str]:
        return self._targets

    @property
    def target_name(self) -> str:
        if len(self._targets) != 1:
            raise ValueError(
                f"The annotation layer has more or less than one target, can not return a single target name: "
                f"{self._targets}"
            )
        return self._targets[0]

    @property
    def target(self) -> Any:
        return self.targets[self.target_name]

    @property
    def target_layers(self) -> dict[str, "AnnotationLayer"]:
        return {
            target_name: target
            for target_name, target in self.targets.items()
            if isinstance(target, AnnotationLayer)
        }

    @property
    def target_layer(self) -> "AnnotationLayer":
        tgt_layers = self.target_layers
        if len(tgt_layers) != 1:
            raise ValueError(
                f"The annotation layer has more or less than one target layer: {list(tgt_layers.keys())}"
            )
        return list(tgt_layers.values())[0]


class AnnotationLayer(BaseAnnotationList[T]):
    def __init__(self, document: "Document", targets: List["str"]):
        super().__init__(document=document, targets=targets)
        self._predictions: BaseAnnotationList[T] = BaseAnnotationList(document, targets=targets)

    @property
    def predictions(self) -> BaseAnnotationList[T]:
        return self._predictions

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AnnotationLayer):
            return NotImplemented

        return super().__eq__(other) and self.predictions == other.predictions

    def __repr__(self) -> str:
        return f"AnnotationLayer({str(self._annotations)})"


D = TypeVar("D", bound="Document")


@dataclasses.dataclass
class Document(Mapping[str, Any]):
    # points from annotation field names to lists of target field names
    _annotation_graph: Dict[str, List[str]] = dataclasses.field(
        default_factory=dict, init=False, repr=False
    )
    _annotation_fields: Set[str] = dataclasses.field(default_factory=set, init=False, repr=False)

    @classmethod
    def fields(cls):
        return [
            f
            for f in dataclasses.fields(cls)
            if f.name not in {"_annotation_graph", "_annotation_fields"}
        ]

    @classmethod
    def field_types(cls) -> Dict[str, typing.Type]:
        result = {}
        for f in cls.fields():
            # If we got just the string representation of the type, we resolve the whole class.
            # But this may be slow, so we only do it if necessary.
            if not isinstance(f.type, type):
                return typing.get_type_hints(cls)
            result[f.name] = f.type
        return result

    @classmethod
    def annotation_fields(cls) -> Set[dataclasses.Field]:
        ann_field_types = cls.field_types()
        return {
            f
            for f in cls.fields()
            if typing.get_origin(ann_field_types[f.name]) is AnnotationLayer
        }

    def __getitem__(self, key: str) -> AnnotationLayer:
        if key not in self._annotation_fields:
            raise KeyError(f"Document has no attribute '{key}'.")
        return getattr(self, key)

    def __iter__(self):
        return iter(self._annotation_fields)

    def __len__(self):
        return len(self._annotation_fields)

    def __post_init__(self):
        targeted = set()
        field_names = {field.name for field in dataclasses.fields(self)}
        field_types = self.field_types()
        for field in self.annotation_fields():

            self._annotation_fields.add(field.name)

            targets = field.metadata.get("targets")
            for target in targets:
                targeted.add(target)
                if field.name not in self._annotation_graph:
                    self._annotation_graph[field.name] = []
                self._annotation_graph[field.name].append(target)
                if target not in field_names:
                    raise TypeError(
                        f'annotation target "{target}" is not in field names of the document: {field_names}'
                    )

            # check annotation target names and use them together with target names from the AnnotationLayer
            # to reorder targets, if available
            target_names = field.metadata.get("target_names")
            field_type = field_types[field.name]
            annotation_type = typing.get_args(field_type)[0]
            annotation_target_names = annotation_type.TARGET_NAMES
            if annotation_target_names is not None:
                if target_names is not None:
                    if set(target_names) != set(annotation_target_names):
                        raise TypeError(
                            f"keys of targets {sorted(target_names)} do not match "
                            f"{annotation_type.__name__}.TARGET_NAMES {sorted(annotation_target_names)}"
                        )
                    # reorder targets according to annotation_target_names
                    target_name_mapping = dict(zip(target_names, targets))
                    target_position_mapping = {
                        i: target_name_mapping[name]
                        for i, name in enumerate(annotation_target_names)
                    }
                    targets = [target_position_mapping[i] for i in range(len(targets))]
                else:
                    if len(annotation_target_names) != len(targets):
                        raise TypeError(
                            f"number of targets {sorted(targets)} does not match number of entries in "
                            f"{annotation_type.__name__}.TARGET_NAMES: {sorted(annotation_target_names)}"
                        )
                    # disallow multiple targets when target names are specified in the definition of the Annotation
                    if len(annotation_target_names) > 1:
                        raise TypeError(
                            f"A target name mapping is required for AnnotationLayers containing Annotations with "
                            f'TARGET_NAMES, but AnnotationLayer "{field.name}" has no target_names. You should '
                            f"pass the named_targets dict containing the following keys (see Annotation "
                            f'"{annotation_type.__name__}") to annotation_field: {annotation_target_names}'
                        )

            field_value = field_type(document=self, targets=targets)
            setattr(self, field.name, field_value)

        if "_artificial_root" in self._annotation_graph:
            raise ValueError(
                'Failed to add the "_artificial_root" node to the annotation graph because it already exists. Note '
                "that AnnotationLayer entries with that name are not allowed."
            )
        self._annotation_graph["_artificial_root"] = list(self._annotation_fields - targeted)

    def asdict(self):
        dct = {}
        for field in self.fields():
            value = getattr(self, field.name)

            if isinstance(value, AnnotationLayer):
                dct[field.name] = {
                    "annotations": [v.asdict() for v in value],
                    "predictions": [v.asdict() for v in value.predictions],
                }
            elif isinstance(value, dict):
                dct[field.name] = value or None
            else:
                dct[field.name] = value

        return dct

    @classmethod
    def fromdict(cls, dct):
        fields = dataclasses.fields(cls)
        annotation_fields = cls.annotation_fields()
        field_types = cls.field_types()

        cls_kwargs = {}
        for field in fields:
            if field not in annotation_fields:
                value = dct.get(field.name)

                if value is not None:
                    cls_kwargs[field.name] = value

        doc = cls(**cls_kwargs)

        name_to_field = {f.name: f for f in annotation_fields}

        dependency_ordered_fields: List[str] = []
        _enumerate_dependencies(
            dependency_ordered_fields,
            dependency_graph=doc._annotation_graph,
            nodes=doc._annotation_graph["_artificial_root"],
        )

        annotations = {}
        predictions = {}
        annotations_per_field = defaultdict(list)
        predictions_per_field = defaultdict(list)
        for field_name in dependency_ordered_fields:
            # terminal nodes do not have to be an annotation field (e.g. the text field)
            if field_name not in name_to_field:
                continue

            field = name_to_field[field_name]

            value = dct.get(field.name)

            if value is None or not value:
                continue

            field_type = field_types[field_name]
            # TODO: handle single annotations, e.g. a document-level label
            if typing.get_origin(field_type) is AnnotationLayer:
                annotation_class = typing.get_args(field_type)[0]
                # build annotations
                for annotation_data in value["annotations"]:
                    annotation_dict = dict(annotation_data)
                    annotation_id = annotation_dict.pop("_id")
                    # annotations can only reference annotations
                    annotation = annotation_class.fromdict(annotation_dict, annotations)
                    annotations[annotation_id] = annotation
                    annotations_per_field[field.name].append(annotation)
                # build predictions
                for annotation_data in value["predictions"]:
                    annotation_dict = dict(annotation_data)
                    annotation_id = annotation_dict.pop("_id")
                    # predictions can reference annotations and predictions
                    annotation = annotation_class.fromdict(
                        annotation_dict, {**annotations, **predictions}
                    )
                    predictions[annotation_id] = annotation
                    predictions_per_field[field.name].append(annotation)
            else:
                raise Exception("Error")

        for field_name, field_annotations in annotations_per_field.items():
            getattr(doc, field_name).extend(field_annotations)

        for field_name, field_annotations in predictions_per_field.items():
            getattr(doc, field_name).predictions.extend(field_annotations)

        return doc

    def as_type(
        self,
        new_type: typing.Type[D],
        field_mapping: Optional[Dict[str, str]] = None,
        keep_remaining: bool = True,
    ) -> D:
        field_mapping = field_mapping or {}
        new_doc = new_type.fromdict(
            {
                field_mapping.get(k, k): v
                for k, v in self.asdict().items()
                if keep_remaining or k in field_mapping
            }
        )
        return new_doc

    def copy(self, with_annotations: bool = True) -> "Document":
        doc_dict = self.asdict()
        if not with_annotations:
            for field in self.annotation_fields():
                doc_dict.pop(field.name)
        return type(self).fromdict(doc_dict)

    def add_all_annotations_from_other(
        self,
        other: "Document",
        removed_annotations: Optional[Dict[str, Set[int]]] = None,
        override_annotations: Optional[Dict[str, Dict[int, Annotation]]] = None,
        process_predictions: bool = True,
        strict: bool = True,
        verbose: bool = True,
    ) -> Dict[str, List[Annotation]]:
        """Adds all annotations from another document to this document. It allows to blacklist annotations
        and also to override annotations. It returns the original annotations for which a new annotation was
        added to the current document.

        The method is useful if e.g. a text-based document is converted to a token-based document and the
        annotations should be added to the token-based document.

        Args:
            other: The document to add annotations from.
            process_predictions: Whether to process predictions as well (default: True). If set to False,
                the predictions in the other document will be ignored.
            override_annotations: A mapping from annotation field names to a mapping from
                annotation IDs to annotations. The effects are two-fold. First, adding any annotation
                field name as key has the effect that the field is expected to be already handled and
                no annotations will be added from the other document. Second, if a certain mapping
                old_annotation._id -> new_annotation is present in the mapping for a certain field,
                the new_annotation will be used anywhere where the old_annotation would have been
                referenced. This propagates along the annotation graph and can be useful if some
                annotations are modified, but all dependent annotations should be kept intact e.g.
                when converting a text-based document to token-based (or the other way around). See below
                for an example.
            removed_annotations: A mapping from annotation field names to a set of annotation ids that
                are removed from the document. This is useful if e.g. the annotation base (the text)
                is split up and the annotations need to be split up as well or if conversion to a token base
                is performed, but some spans could not be converted, because of aligning issues, and, thus,
                should be removed. Similar as for entries in override_annotations, fields that are mentioned
                in removed_annotations are expected to be already handled manually and will not be added from
                the other document. Also, the removal of annotations propagates along the annotation graph.
            strict: Whether to raise an exception if the other document contains annotations that reference
                annotations that are not present in the current document (see parameter removed_annotations).
                If set to False, the annotations are ignored.
            verbose: Whether to print a warning if the other document contains annotations that reference
                annotations that are not present in the current document (see parameter removed_annotations).

        Returns:
            A mapping from annotation field names to the set of annotations from the original document for which
            a new annotation was added to the current document. This can be useful to check if all original
            annotations were added (possibly to multiple target documents).

        Example:
                ```
                @dataclasses.dataclass(frozen=True)
                class Attribute(Annotation):
                    ref: Annotation
                    value: str

                @dataclasses.dataclass
                class TextBasedDocumentWithEntitiesRelationsAndRelationAttributes(TextBasedDocument):
                    entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
                    relations: AnnotationLayer[BinaryRelation] = annotation_field(target="entities")
                    relation_attributes: AnnotationLayer[Attribute] = annotation_field(target="relations")

                @dataclasses.dataclass
                class TokenBasedDocumentWithEntitiesRelationsAndRelationAttributes(TokenBasedDocument):
                    entities: AnnotationLayer[LabeledSpan] = annotation_field(target="tokens")
                    relations: AnnotationLayer[BinaryRelation] = annotation_field(target="entities")
                    relation_attributes: AnnotationLayer[Attribute] = annotation_field(target="relations")


                doc_text = TextBasedDocumentWithEntitiesRelationsAndRelationAttributes(text="Hello World!")
                e1_text = LabeledSpan(0, 5, "word1")
                e2_text = LabeledSpan(6, 11, "word2")
                doc_text.entities.extend([e1, e2])
                r1_text = BinaryRelation(e1_text, e2_text, "relation1")
                doc_text.relations.append(r1_text)
                doc_text.relation_attributes.append(Attribute(r1_text, "attribute1"))

                doc_tokens = TokenBasedDocumentWithEntitiesRelationsAndRelationAttributes(
                    tokens=("Hello", "World", "!")
                )
                e1_tokens = LabeledSpan(0, 1, "word1")
                e2_tokens = LabeledSpan(1, 2, "word2")
                doc_tokens.entities.extend([e1_tokens, e2_tokens])
                doc_tokens.add_all_annotations_from_other(
                    other=doc_text,
                    override_annotations={"entities": {e1_text._id: e1_tokens, e2_text._id: e2_tokens}},
                )
                # Note that the relation and attribute are still present, but now refer to the new entities
                # and new relation, respectively.
                ```
        """
        removed_annotations = defaultdict(set, removed_annotations or dict())
        added_annotations = defaultdict(list)

        annotation_store: Dict[str, Dict[int, Annotation]] = defaultdict(dict)
        named_annotation_fields = {field.name: field for field in self.annotation_fields()}
        if override_annotations is not None:
            for field_name, mapping in override_annotations.items():
                if field_name not in named_annotation_fields:
                    raise ValueError(
                        f'Field "{field_name}" is not an annotation field of {type(self).__name__}, but keys in '
                        f"override_annotation_mapping must be annotation field names."
                    )
                annotation_store[field_name].update(mapping)
        else:
            override_annotations = dict()

        dependency_ordered_fields: List[str] = []
        _enumerate_dependencies(
            dependency_ordered_fields,
            dependency_graph=self._annotation_graph,
            nodes=self._annotation_graph["_artificial_root"],
        )
        for field_name in dependency_ordered_fields:
            # we process only annotation fields that are not in the override_annotations and the removed_annotations
            # mapping because they are meant to be already manually handled
            if field_name in named_annotation_fields and field_name not in (
                set(override_annotations) | set(removed_annotations)
            ):
                current_targets_store = dict()
                current_invalid_annotation_ids = set()
                for target in self._annotation_graph.get(field_name, []):
                    current_targets_store.update(annotation_store[target])
                    current_invalid_annotation_ids.update(removed_annotations.get(target, set()))

                other_annotation_field = other[field_name]
                for ann in other_annotation_field:
                    new_ann = ann.copy_with_store(
                        override_annotation_store=current_targets_store,
                        invalid_annotation_ids=current_invalid_annotation_ids,
                    )
                    if new_ann is not None:
                        if ann._id != new_ann._id:
                            annotation_store[field_name][ann._id] = new_ann
                        self[field_name].append(new_ann)
                        added_annotations[field_name].append(ann)
                    else:
                        if strict:
                            raise ValueError(
                                f"Could not add annotation {ann} to {type(self).__name__} because it depends on "
                                f"annotations that are not present in the document."
                            )
                        if verbose:
                            logger.warning(
                                f"Could not add annotation {ann} to {type(self).__name__} because it depends on "
                                f"annotations that are not present in the document. The annotation is ignored."
                                f"(disable this warning with verbose=False)"
                            )
                        # The annotation was removed, so we need to make sure that it is not referenced by any other
                        removed_annotations[field_name].add(ann._id)
                if process_predictions:
                    for ann in other_annotation_field.predictions:
                        new_ann = ann.copy_with_store(
                            override_annotation_store=current_targets_store,
                            invalid_annotation_ids=current_invalid_annotation_ids,
                        )
                        if new_ann is not None:
                            if ann._id != new_ann._id:
                                annotation_store[field_name][ann._id] = new_ann
                            self[field_name].predictions.append(new_ann)
                            added_annotations[field_name].append(ann)
                        else:
                            if strict:
                                raise ValueError(
                                    f"Could not add annotation {ann} to {type(self).__name__} because it depends on "
                                    f"annotations that are not present in the document."
                                )
                            if verbose:
                                logger.warning(
                                    f"Could not add annotation {ann} to {type(self).__name__} because it depends on "
                                    f"annotations that are not present in the document. The annotation is ignored. "
                                    f"(disable this warning with verbose=False)"
                                )
                            # The annotation was removed, so we need to make sure that it is not referenced by any other
                            removed_annotations[field_name].add(ann._id)

        return dict(added_annotations)


def resolve_annotation(
    id_or_annotation: Union[int, Annotation],
    store: Optional[Dict[int, Annotation]],
) -> Annotation:
    if isinstance(id_or_annotation, Annotation):
        return id_or_annotation
    else:
        if store is None:
            raise ValueError("Unable to resolve the annotation id without annotation_store.")
        return store[id_or_annotation]
