import dataclasses
import json
from typing import Dict, List, Optional, Tuple

import pytest

from pytorch_ie.annotations import BinaryRelation
from pytorch_ie.core import Annotation
from pytorch_ie.core.document import (
    AnnotationLayer,
    Document,
    _contains_annotation_type,
    _get_reference_fields_and_container_types,
    _is_annotation_type,
    _is_optional_annotation_type,
    _is_optional_type,
    _is_tuple_of_annotation_types,
    annotation_field,
)


@dataclasses.dataclass(eq=True, frozen=True)
class Span(Annotation):
    start: int
    end: int


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
        score: float = dataclasses.field(default=1.0, compare=False)

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
        score: float = dataclasses.field(default=1.0, compare=False)

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


def test_annotation_sort():
    @dataclasses.dataclass(eq=True, frozen=True)
    class Dummy(Annotation):
        a: int
        b: Optional[int] = None
        c: int = 0

    dummy1 = Dummy(a=1, c=2)
    dummy2 = Dummy(a=1, c=3)
    dummy3 = Dummy(a=2, c=1)
    dummy4 = Dummy(a=2, c=2)

    assert sorted([dummy1, dummy2, dummy3, dummy4]) == [dummy1, dummy2, dummy3, dummy4]

    @dataclasses.dataclass(eq=True, frozen=True)
    class DummyWithNestedAnnotation(Annotation):
        a: int
        n: Dummy

    dummy_nested1 = DummyWithNestedAnnotation(a=1, n=Dummy(a=1, c=2))
    dummy_nested2 = DummyWithNestedAnnotation(a=2, n=Dummy(a=2, c=3))
    dummy_nested3 = DummyWithNestedAnnotation(a=2, n=Dummy(a=1, c=4))
    dummy_nested4 = DummyWithNestedAnnotation(a=1, n=Dummy(a=2, c=2))

    assert sorted([dummy_nested1, dummy_nested2, dummy_nested3, dummy_nested4]) == [
        dummy_nested1,
        dummy_nested4,
        dummy_nested3,
        dummy_nested2,
    ]

    with pytest.raises(ValueError) as excinfo:
        sorted([dummy1, dummy_nested1])
    assert (
        str(excinfo.value)
        == "comparison field names do not match: ['_targets', 'a', 'n'] != ['_targets', 'a', 'c', 'b']"
    )


def test_annotation_is_attached():
    @dataclasses.dataclass
    class MyDocument(Document):
        text: str
        words: AnnotationLayer[Span] = annotation_field(target="text")

    document = MyDocument(text="Hello world!")
    word = Span(start=0, end=5)
    assert not word.is_attached
    document.words.append(word)
    assert word.is_attached
    document.words.pop()
    assert not word.is_attached


def test_annotation_copy():
    @dataclasses.dataclass(eq=True, frozen=True)
    class Attribute(Annotation):
        annotation: Annotation
        label: str

        def __repr__(self):
            return f"Attribute(annotation={self.annotation}, label={self.label})"

    @dataclasses.dataclass
    class MyDocument(Document):
        text: str
        words: AnnotationLayer[Span] = annotation_field(target="text")
        attributes: AnnotationLayer[Attribute] = annotation_field(target="words")

    document = MyDocument(text="Hello world!")
    word = Span(start=0, end=5)
    attribute = Attribute(annotation=word, label="label")
    # both annotations are not yet attached
    assert not word.is_attached
    assert not attribute.is_attached
    # copy the annotations
    attribute_copy0 = attribute.copy()
    word_copy0 = word.copy()
    # now attach the annotations
    document.words.append(word)
    document.attributes.append(attribute)
    assert word.is_attached
    assert attribute.is_attached
    # copy the annotations again
    word_copy1 = word.copy()
    attribute_copy1 = attribute.copy()
    # check that the copies are not attached
    assert not word_copy1.is_attached
    assert not attribute_copy1.is_attached
    # check that the copies have the same values as the originals
    assert word_copy1.start == word.start
    assert word_copy1.end == word.end
    assert attribute_copy1.annotation == attribute.annotation
    assert attribute_copy1.label == attribute.label
    # check that the copies before attaching the originals are the same as the copies after attaching the originals
    assert word_copy1 == word_copy0
    assert attribute_copy1 == attribute_copy0

    # create a copy of the attribute, but let it point to a new word, i.e. overwrite a field
    new_word = Span(start=6, end=11)
    document.words.append(new_word)
    attribute_copy2 = attribute.copy(annotation=new_word)
    document.attributes.append(attribute_copy2)
    assert len(document.attributes) == 2
    assert str(document.attributes[0]) == "Attribute(annotation=Span(start=0, end=5), label=label)"
    assert (
        str(document.attributes[1]) == "Attribute(annotation=Span(start=6, end=11), label=label)"
    )


def test_document_annotation_fields():
    @dataclasses.dataclass
    class MyDocument(Document):
        text: str
        words: AnnotationLayer[Span] = annotation_field(target="text")

    annotation_fields = MyDocument.annotation_fields()
    annotation_field_names = {field.name for field in annotation_fields}
    assert annotation_field_names == {"words"}


def test_document_target_names():
    @dataclasses.dataclass
    class MyDocument(Document):
        text: str
        words: AnnotationLayer[Span] = annotation_field(target="text")
        sentences: AnnotationLayer[Span] = annotation_field(target="text")
        belongs_to: AnnotationLayer[BinaryRelation] = annotation_field(
            targets=["words", "sentences"]
        )

    # request target names for annotation field
    assert MyDocument.target_names("words") == {"text"}
    assert MyDocument.target_name("words") == "text"

    # requested field is not an annotation field
    with pytest.raises(ValueError) as excinfo:
        MyDocument.target_names("text")
    assert str(excinfo.value) == f"'text' is not an annotation field of {MyDocument.__name__}."

    # requested field has two targets
    assert MyDocument.target_names("belongs_to") == {"words", "sentences"}
    with pytest.raises(ValueError) as excinfo:
        MyDocument.target_name("belongs_to")
    assert (
        str(excinfo.value)
        == f"The annotation field 'belongs_to' has more or less than one target, can not return a single target name: ['sentences', 'words']"
    )


def test_document_copy():
    @dataclasses.dataclass
    class MyDocument(Document):
        text: str
        words: AnnotationLayer[Span] = annotation_field(target="text")

    document = MyDocument(text="Hello world!")
    word = Span(start=0, end=5)
    document.words.append(word)
    document_copy = document.copy()
    assert document_copy == document

    # copy without annotations
    document_copy = document.copy(with_annotations=False)
    assert document_copy != document
    annotation_fields = document_copy.annotation_fields()
    assert len(annotation_fields) > 0
    for field in dataclasses.fields(document):
        if field in annotation_fields:
            assert getattr(document_copy, field.name) != getattr(document, field.name)
        else:
            assert getattr(document_copy, field.name) == getattr(document, field.name)


def test_document_target_name_and_target():
    @dataclasses.dataclass
    class MyDocument(Document):
        text: str
        words: AnnotationLayer[Span] = annotation_field(target="text")

    document = MyDocument(text="Hello world!")
    assert document.words.target_name == "text"
    assert document.words.target == document.text == "Hello world!"

    class DoubleSpan(Annotation):
        start1: int
        end1: int
        start2: int
        end2: int

    @dataclasses.dataclass
    class MyDocumentTwoTargets(Document):
        text1: str
        text2: str
        words: AnnotationLayer[DoubleSpan] = annotation_field(targets=["text1", "text2"])

    document = MyDocumentTwoTargets(text1="Hello world!", text2="Hello world again!")
    with pytest.raises(ValueError) as excinfo:
        document.words.target_name
    assert (
        str(excinfo.value)
        == "The annotation layer has more or less than one target, can not return a single target name: "
        "['text1', 'text2']"
    )
    with pytest.raises(ValueError) as excinfo:
        document.words.target
    assert (
        str(excinfo.value)
        == "The annotation layer has more or less than one target, can not return a single target name: "
        "['text1', 'text2']"
    )

    assert document.words.target_names == ["text1", "text2"]
    assert document.words.targets == {"text1": "Hello world!", "text2": "Hello world again!"}
