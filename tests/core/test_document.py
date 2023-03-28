import dataclasses
from typing import List, Optional, Tuple

import pytest

from pytorch_ie.annotations import Span
from pytorch_ie.core import Annotation
from pytorch_ie.core.document import (
    _contains_annotation_type,
    _get_reference_fields_and_container_types,
    _is_annotation_subclass,
    _is_optional_type,
    _is_tuple_of_annotations,
)


def test_is_optional_type():
    @dataclasses.dataclass(eq=True, frozen=True)
    class Dummy:
        a: int
        b: Tuple[int, ...]
        c: Optional[List[int]]
        d: Optional[int] = None

    fields = {field.name: field for field in dataclasses.fields(Dummy)}
    assert not _is_optional_type(fields["a"].type)
    assert not _is_optional_type(fields["b"].type)
    assert _is_optional_type(fields["c"].type)
    assert _is_optional_type(fields["d"].type)


def test_is_annotation_subclass():
    @dataclasses.dataclass(eq=True, frozen=True)
    class Dummy:
        a: Span
        b: Annotation
        c: Optional[List[int]]
        d: Optional[int] = None

    fields = {field.name: field for field in dataclasses.fields(Dummy)}
    assert _is_annotation_subclass(fields["a"].type)
    assert _is_annotation_subclass(fields["b"].type)
    assert not _is_annotation_subclass(fields["c"].type)
    assert not _is_annotation_subclass(fields["d"].type)


def test_contains_annotation_type():
    @dataclasses.dataclass(eq=True, frozen=True)
    class Dummy:
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


def test_is_tuple_of_annotations():
    @dataclasses.dataclass(eq=True, frozen=True)
    class Dummy:
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
    assert not _is_tuple_of_annotations(
        fields["a"].type
    ), f'field "a" is a pure tuple of annotation type'
    assert not _is_tuple_of_annotations(
        fields["b"].type
    ), f'field "b" is a pure tuple of annotation type'
    assert not _is_tuple_of_annotations(
        fields["c"].type
    ), f'field "c" is a pure tuple of annotation type'
    assert _is_tuple_of_annotations(
        fields["d"].type
    ), f'field "d" is not a pure tuple of annotation type'
    assert not _is_tuple_of_annotations(
        fields["e"].type
    ), f'field "e" is a pure tuple of annotation type'
    assert _is_tuple_of_annotations(
        fields["f"].type
    ), f'field "f" is not a pure tuple of annotation type'
    assert _is_tuple_of_annotations(
        fields["g"].type
    ), f'field "g" is not a pure tuple of annotation type'
    with pytest.raises(TypeError):
        _is_tuple_of_annotations(fields["h"].type)
    assert _is_tuple_of_annotations(
        fields["i"].type
    ), f'field "i" is not a pure tuple of annotation type'


def test_get_annotation_fields_with_container_exception():
    @dataclasses.dataclass(eq=True, frozen=True)
    class Dummy:
        a: Annotation
        # This causes an exception because the type of by contains an Annotation subclass (Span),
        # but it is embedded *twice* in a Tuple which is not allowed.
        b: Tuple[Tuple[Span, ...]]

    with pytest.raises(TypeError):
        _get_reference_fields_and_container_types(Dummy)
