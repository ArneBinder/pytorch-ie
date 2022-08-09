import dataclasses
import typing
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, TypeVar, Union, overload


def _depth_first_search(lst: List[str], visited: Set[str], graph: Dict[str, List[str]], node: str):
    if node not in visited:
        lst.append(node)
        visited.add(node)
        neighbours: List[str] = graph.get(node) or []
        for neighbour in neighbours:
            _depth_first_search(lst, visited, graph, neighbour)


def _get_annotation_fields(fields: List[dataclasses.Field]) -> Set[dataclasses.Field]:
    return {field for field in fields if typing.get_origin(field.type) is AnnotationList}


def annotation_field(target: Optional[str] = None):
    return dataclasses.field(metadata=dict(target=target), init=False, repr=False)


@dataclasses.dataclass(eq=True, frozen=True)
class Annotation:
    _target: Optional[Union["AnnotationList", str]] = dataclasses.field(
        default=None, init=False, repr=False, hash=False
    )

    def set_target(self, value: Union["AnnotationList", str, None]):
        object.__setattr__(self, "_target", value)

    @property
    def target(self) -> Optional[Union["AnnotationList", str]]:
        return self._target

    def asdict(self) -> Dict[str, Any]:
        dct = dataclasses.asdict(self)
        dct["_id"] = hash(self)
        del dct["_target"]
        return dct

    @classmethod
    def fromdict(
        cls,
        dct: Dict[str, Any],
        annotation_store: Optional[Dict[int, Tuple[str, "Annotation"]]] = None,
    ):
        tmp_dct = dict(dct)
        tmp_dct.pop("_id", None)
        return cls(**tmp_dct)


T = TypeVar("T", covariant=False, bound="Annotation")


class BaseAnnotationList(Sequence[T]):
    def __init__(self, document: "Document", target: "str"):
        self._document = document
        self._target = target
        self._annotations: List[T] = []

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseAnnotationList):
            return NotImplemented

        return self._target == other._target and self._annotations == other._annotations

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
        annotation.set_target(getattr(self._document, self._target))
        self._annotations.append(annotation)

    def extend(self, annotations: Iterable[T]) -> None:
        for annotation in annotations:
            self.append(annotation)

    def __repr__(self) -> str:
        return f"BaseAnnotationList({str(self._annotations)})"

    def clear(self):
        for annotation in self._annotations:
            annotation.set_target(None)
        self._annotations = []


class AnnotationList(BaseAnnotationList[T]):
    def __init__(self, document: "Document", target: "str"):
        super().__init__(document=document, target=target)
        self._predictions: BaseAnnotationList[T] = BaseAnnotationList(document, target)

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
    _root_annotation: Optional[str] = dataclasses.field(default=None, init=False, repr=False)

    def __getitem__(self, key: str) -> AnnotationList:
        if key not in self._annotation_fields:
            raise KeyError(f"Document has no attribute '{key}'.")
        return getattr(self, key)

    def __iter__(self):
        return iter(self._annotation_fields)

    def __len__(self):
        return len(self._annotation_fields)

    def __post_init__(self):
        edges = set()
        for field in dataclasses.fields(self):
            if field.name == "_annotation_graph":
                continue

            field_origin = typing.get_origin(field.type)

            if field_origin is AnnotationList:
                self._annotation_fields.add(field.name)

                annotation_target = field.metadata.get("target")
                edges.add((field.name, annotation_target))
                field_value = field.type(document=self, target=annotation_target)
                setattr(self, field.name, field_value)

        for edge in edges:
            src, dst = edge
            if dst not in self._annotation_graph:
                self._annotation_graph[dst] = []
            self._annotation_graph[dst].append(src)

    def asdict(self):
        dct = {}
        for field in dataclasses.fields(self):
            if field.name in {"_annotation_graph", "_annotation_fields", "_root_annotation"}:
                continue

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

        dependency_ordered_fields: List[dataclasses.Field] = []

        _depth_first_search(
            lst=dependency_ordered_fields,
            visited=set(),
            graph=doc._annotation_graph,
            node=doc._root_annotation,
        )

        annotations = {}
        predictions = {}
        for field_name in dependency_ordered_fields:
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
                    annotations[annotation_id] = (
                        field.name,
                        # annotations can only reference annotations
                        annotation_class.fromdict(annotation_dict, annotations),
                    )
                # build predictions
                for annotation_data in value["predictions"]:
                    annotation_dict = dict(annotation_data)
                    annotation_id = annotation_dict.pop("_id")
                    predictions[annotation_id] = (
                        field.name,
                        # predictions can reference annotations and predictions
                        annotation_class.fromdict(annotation_dict, {**annotations, **predictions}),
                    )
            else:
                raise Exception("Error")

        for field_name, annotation in annotations.values():
            getattr(doc, field_name).append(annotation)

        for field_name, annotation in predictions.values():
            getattr(doc, field_name).predictions.append(annotation)

        return doc

    def as_type(self, new_type: typing.Type[D], field_mapping: Optional[Dict[str, str]] = None):
        field_mapping = field_mapping or {}
        new_doc = new_type.fromdict({field_mapping.get(k, k): v for k, v in self.asdict().items()})
        return new_doc
