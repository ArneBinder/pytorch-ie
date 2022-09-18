import dataclasses
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

from typing_extensions import SupportsIndex, TypeAlias


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


def _get_annotation_fields(fields: List[dataclasses.Field]) -> Set[dataclasses.Field]:
    return {field for field in fields if typing.get_origin(field.type) is AnnotationList}


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
TARGET_TYPE = Union["AnnotationList", str]


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
    def _id(self) -> int:
        return hash(self)

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
        return self._asdict()

    @classmethod
    def fromdict(
        cls,
        dct: Dict[str, Any],
        annotation_store: Optional[Dict[int, "Annotation"]] = None,
    ):
        tmp_dct = dict(dct)
        tmp_dct.pop("_id", None)
        return cls(**tmp_dct)


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

    def clear(self):
        for annotation in self._annotations:
            annotation.set_targets(None)
        self._annotations = []

    def pop(self, index=None):
        ann = self._annotations.pop(index)
        ann.set_targets(None)
        return ann


class AnnotationList(BaseAnnotationList[T]):
    def __init__(self, document: "Document", targets: List["str"]):
        super().__init__(document=document, targets=targets)
        self._predictions: BaseAnnotationList[T] = BaseAnnotationList(document, targets=targets)

    @property
    def predictions(self) -> BaseAnnotationList[T]:
        return self._predictions

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AnnotationList):
            return NotImplemented

        return super().__eq__(other) and self.predictions == other.predictions

    def __repr__(self) -> str:
        return f"AnnotationList({str(self._annotations)})"


D = TypeVar("D", bound="Document")


@dataclasses.dataclass
class Document(Mapping[str, Any]):
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
    def annotation_fields(cls):
        return _get_annotation_fields(list(dataclasses.fields(cls)))

    def __getitem__(self, key: str) -> AnnotationList:
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
        for field in dataclasses.fields(self):
            if field.name == "_annotation_graph":
                continue

            field_origin = typing.get_origin(field.type)

            if field_origin is AnnotationList:
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

                # check annotation target names and use them together with target names from the AnnotationList
                # to reorder targets, if available
                target_names = field.metadata.get("target_names")
                annotation_type = typing.get_args(field.type)[0]
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
                                f"A target name mapping is required for AnnotationLists containing Annotations with "
                                f'TARGET_NAMES, but AnnotationList "{field.name}" has no target_names. You should '
                                f"pass the named_targets dict containing the following keys (see Annotation "
                                f'"{annotation_type.__name__}") to annotation_field: {annotation_target_names}'
                            )

                field_value = field.type(document=self, targets=targets)
                setattr(self, field.name, field_value)

        if "_artificial_root" in self._annotation_graph:
            raise ValueError(
                'Failed to add the "_artificial_root" node to the annotation graph because it already exists. Note '
                "that AnnotationList entries with that name are not allowed."
            )
        self._annotation_graph["_artificial_root"] = list(self._annotation_fields - targeted)

    def asdict(self):
        dct = {}
        for field in self.fields():

            value = getattr(self, field.name)

            if isinstance(value, AnnotationList):
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
        annotation_fields = _get_annotation_fields(fields)

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

            # TODO: handle single annotations, e.g. a document-level label
            if typing.get_origin(field.type) is AnnotationList:
                annotation_class = typing.get_args(field.type)[0]
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
        self, new_type: typing.Type[D], field_mapping: Optional[Dict[str, str]] = None
    ) -> D:
        field_mapping = field_mapping or {}
        new_doc = new_type.fromdict({field_mapping.get(k, k): v for k, v in self.asdict().items()})
        return new_doc


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
