import dataclasses
import json
from typing import Dict, List, Optional, Tuple

import pytest

from pytorch_ie.annotations import Span, _post_init_single_label
from pytorch_ie.core import Annotation
from pytorch_ie.core.document import (
    _contains_annotation_type,
    _get_reference_fields_and_container_types,
    _is_annotation_type,
    _is_optional_annotation_type,
    _is_optional_type,
    _is_tuple_of_annotation_types,
)


def _test_annotation_reconstruction(
    annotation: Annotation, annotation_store: Optional[Dict[int, Annotation]] = None
):
    ann_str = json.dumps(annotation.asdict())
    annotation_reconstructed = type(annotation).fromdict(
        json.loads(ann_str), annotation_store=annotation_store
    )
    assert annotation_reconstructed == annotation


def test_is_optional_type():
    @dataclasses.dataclass(eq=True, frozen=True)
    class Dummy(Annotation):
        a: int
        b: Tuple[int, ...]
        c: Optional[List[int]]
        d: Optional[int] = None

    fields = {field.name: field for field in dataclasses.fields(Dummy)}
    assert not _is_optional_type(fields["a"].type)
    assert not _is_optional_type(fields["b"].type)
    assert _is_optional_type(fields["c"].type)
    assert _is_optional_type(fields["d"].type)


def test_is_optional_annotation_type():
    @dataclasses.dataclass(eq=True, frozen=True)
    class Dummy(Annotation):
        a: int
        b: Optional[int]
        c: Optional[Span]
        d: Optional[Tuple[Span, ...]] = None

    fields = {field.name: field for field in dataclasses.fields(Dummy)}
    assert not _is_optional_annotation_type(fields["a"].type)
    assert not _is_optional_annotation_type(fields["b"].type)
    assert _is_optional_annotation_type(fields["c"].type)
    assert not _is_optional_annotation_type(fields["d"].type)


def test_is_annotation_type():
    @dataclasses.dataclass(eq=True, frozen=True)
    class Dummy(Annotation):
        a: Span
        b: Annotation
        c: Optional[List[int]]
        d: Optional[int] = None

    fields = {field.name: field for field in dataclasses.fields(Dummy)}
    assert _is_annotation_type(fields["a"].type)
    assert _is_annotation_type(fields["b"].type)
    assert not _is_annotation_type(fields["c"].type)
    assert not _is_annotation_type(fields["d"].type)


def test_contains_annotation_type():
    @dataclasses.dataclass(eq=True, frozen=True)
    class Dummy(Annotation):
        a: Span
        b: Annotation
        c: int
        d: Optional[List[Tuple[Optional[Span], ...]]]
        e: Optional[int] = None

    fields = {field.name: field for field in dataclasses.fields(Dummy)}
    assert _contains_annotation_type(
        fields["a"].type
    ), f'field "a" does not contain an annotation type'
    assert _contains_annotation_type(
        fields["b"].type
    ), f'field "b" does not contain an annotation type'
    assert not _contains_annotation_type(
        fields["c"].type
    ), f'field "c" does contain an annotation type'
    assert _contains_annotation_type(
        fields["d"].type
    ), f'field "d" does not contain an annotation type'
    assert not _contains_annotation_type(
        fields["e"].type
    ), f'field "e" does not contain an annotation type'


def test_is_tuple_of_annotation_types():
    @dataclasses.dataclass(eq=True, frozen=True)
    class Dummy(Annotation):
        # a: no
        a: Annotation
        # b: no
        b: int
        # c: no, contains optional elements
        c: Tuple[Optional[Span], ...]
        # d: yes
        d: Tuple[Span, ...]
        # e: no, is optional
        e: Optional[Tuple[Span, ...]]
        # f: yes
        f: Tuple[Span, Span]
        # g: yes
        g: Tuple[Span, ...]
        # h: raise exception because it is mixed with non-Annotation type
        h: Tuple[Span, int]
        # i: raise no exception because it is mixed just with Annotation type subclasses
        i: Tuple[Span, Annotation]

    fields = {field.name: field for field in dataclasses.fields(Dummy)}
    assert not _is_tuple_of_annotation_types(
        fields["a"].type
    ), f'field "a" is a pure tuple of annotation type'
    assert not _is_tuple_of_annotation_types(
        fields["b"].type
    ), f'field "b" is a pure tuple of annotation type'
    assert not _is_tuple_of_annotation_types(
        fields["c"].type
    ), f'field "c" is a pure tuple of annotation type'
    assert _is_tuple_of_annotation_types(
        fields["d"].type
    ), f'field "d" is not a pure tuple of annotation type'
    assert not _is_tuple_of_annotation_types(
        fields["e"].type
    ), f'field "e" is a pure tuple of annotation type'
    assert _is_tuple_of_annotation_types(
        fields["f"].type
    ), f'field "f" is not a pure tuple of annotation type'
    assert _is_tuple_of_annotation_types(
        fields["g"].type
    ), f'field "g" is not a pure tuple of annotation type'
    with pytest.raises(TypeError):
        _is_tuple_of_annotation_types(fields["h"].type)
    assert _is_tuple_of_annotation_types(
        fields["i"].type
    ), f'field "i" is not a pure tuple of annotation type'


def test_get_reference_fields_and_container_types():
    @dataclasses.dataclass(eq=True, frozen=True)
    class Dummy(Annotation):
        a: Annotation
        # This causes an exception because the type of by contains an Annotation subclass (Span),
        # but it is embedded *twice* in a Tuple which is not allowed.
        b: Tuple[Tuple[Span, ...]]

    with pytest.raises(TypeError):
        _get_reference_fields_and_container_types(Dummy)


def test_annotation_with_optional_reference():
    @dataclasses.dataclass(eq=True, frozen=True)
    class BinaryRelationWithOptionalTrigger(Annotation):
        head: Span
        tail: Span
        label: str
        trigger: Optional[Span] = None
        score: float = 1.0

        def __post_init__(self) -> None:
            _post_init_single_label(self)

    head = Span(start=1, end=2)
    tail = Span(start=3, end=4)
    trigger = Span(start=5, end=7)

    binary_relation1 = BinaryRelationWithOptionalTrigger(head=head, tail=tail, label="label1")
    assert binary_relation1.head == head
    assert binary_relation1.tail == tail
    assert binary_relation1.label == "label1"
    assert binary_relation1.score == pytest.approx(1.0)

    assert binary_relation1.asdict() == {
        "_id": binary_relation1._id,
        "head": head._id,
        "tail": tail._id,
        "trigger": None,
        "label": "label1",
        "score": 1.0,
    }

    binary_relation2 = BinaryRelationWithOptionalTrigger(
        head=head, tail=tail, label="label2", score=0.5, trigger=trigger
    )
    assert binary_relation2.head == head
    assert binary_relation2.tail == tail
    assert binary_relation2.trigger == trigger
    assert binary_relation2.label == "label2"
    assert binary_relation2.score == pytest.approx(0.5)

    assert binary_relation2.asdict() == {
        "_id": binary_relation2._id,
        "head": head._id,
        "tail": tail._id,
        "trigger": trigger._id,
        "label": "label2",
        "score": 0.5,
    }

    annotation_store = {
        head._id: head,
        tail._id: tail,
        trigger._id: trigger,
    }
    _test_annotation_reconstruction(binary_relation1, annotation_store=annotation_store)
    _test_annotation_reconstruction(binary_relation2, annotation_store=annotation_store)


def test_annotation_with_tuple_of_references():
    @dataclasses.dataclass(eq=True, frozen=True)
    class BinaryRelationWithEvidence(Annotation):
        head: Span
        tail: Span
        label: str
        evidence: Tuple[Span, ...]
        score: float = 1.0

        def __post_init__(self) -> None:
            _post_init_single_label(self)

    head = Span(start=1, end=2)
    tail = Span(start=3, end=4)
    evidence1 = Span(start=5, end=7)
    evidence2 = Span(start=9, end=10)

    relation = BinaryRelationWithEvidence(
        head=head, tail=tail, label="label1", evidence=(evidence1, evidence2)
    )
    assert relation.head == head
    assert relation.tail == tail
    assert relation.label == "label1"
    assert relation.score == pytest.approx(1.0)
    assert relation.evidence == (evidence1, evidence2)

    assert relation.asdict() == {
        "_id": relation._id,
        "head": head._id,
        "tail": tail._id,
        "evidence": [evidence1._id, evidence2._id],
        "label": "label1",
        "score": 1.0,
    }

    annotation_store = {
        head._id: head,
        tail._id: tail,
        evidence1._id: evidence1,
        evidence2._id: evidence2,
    }
    _test_annotation_reconstruction(relation, annotation_store=annotation_store)
