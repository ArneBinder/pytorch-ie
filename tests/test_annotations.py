import dataclasses
import re

import pytest
from pie_core import AnnotationLayer, annotation_field

from pytorch_ie.annotations import (
    BinaryRelation,
    Label,
    LabeledMultiSpan,
    LabeledSpan,
    MultiLabel,
    MultiLabeledBinaryRelation,
    MultiLabeledSpan,
    NaryRelation,
    Span,
)
from pytorch_ie.documents import TextBasedDocument
from tests.conftest import _test_annotation_reconstruction


def test_label():
    label1 = Label(label="label1")
    assert label1.label == "label1"
    assert label1.score == pytest.approx(1.0)
    assert label1.resolve() == "label1"

    label2 = Label(label="label2", score=0.5)
    assert label2.label == "label2"
    assert label2.score == pytest.approx(0.5)

    assert label2.asdict() == {
        "_id": label2._id,
        "label": "label2",
        "score": 0.5,
    }

    _test_annotation_reconstruction(label2)


def test_multilabel():
    multilabel1 = MultiLabel(label=("label1", "label2"))
    assert multilabel1.label == ("label1", "label2")
    assert multilabel1.score == pytest.approx((1.0, 1.0))
    assert multilabel1.resolve() == ("label1", "label2")

    multilabel2 = MultiLabel(label=("label3", "label4"), score=(0.4, 0.5))
    assert multilabel2.label == ("label3", "label4")
    assert multilabel2.score == pytest.approx((0.4, 0.5))

    assert multilabel2.asdict() == {
        "_id": multilabel2._id,
        "label": ("label3", "label4"),
        "score": (0.4, 0.5),
    }

    _test_annotation_reconstruction(multilabel2)

    with pytest.raises(
        ValueError, match=re.escape("Number of labels (2) and scores (3) must be equal.")
    ):
        MultiLabel(label=("label5", "label6"), score=(0.1, 0.2, 0.3))


def test_span():
    span = Span(start=1, end=2)
    assert span.start == 1
    assert span.end == 2

    assert span.asdict() == {
        "_id": span._id,
        "start": 1,
        "end": 2,
    }

    _test_annotation_reconstruction(span)

    with pytest.raises(ValueError) as excinfo:
        span.resolve()
    assert str(excinfo.value) == "Span(start=1, end=2) is not attached to a target."

    @dataclasses.dataclass
    class TestDocument(TextBasedDocument):
        spans: AnnotationLayer[Span] = annotation_field(target="text")

    doc = TestDocument(text="Hello, world!")
    span = Span(start=7, end=12)
    doc.spans.append(span)
    assert span.resolve() == "world"


def test_labeled_span():
    labeled_span1 = LabeledSpan(start=1, end=2, label="label1")
    assert labeled_span1.start == 1
    assert labeled_span1.end == 2
    assert labeled_span1.label == "label1"
    assert labeled_span1.score == pytest.approx(1.0)

    labeled_span2 = LabeledSpan(start=3, end=4, label="label2", score=0.5)
    assert labeled_span2.start == 3
    assert labeled_span2.end == 4
    assert labeled_span2.label == "label2"
    assert labeled_span2.score == pytest.approx(0.5)

    assert labeled_span2.asdict() == {
        "_id": labeled_span2._id,
        "start": 3,
        "end": 4,
        "label": "label2",
        "score": 0.5,
    }

    _test_annotation_reconstruction(labeled_span2)

    with pytest.raises(ValueError) as excinfo:
        labeled_span1.resolve()
    assert (
        str(excinfo.value)
        == "LabeledSpan(start=1, end=2, label='label1', score=1.0) is not attached to a target."
    )

    @dataclasses.dataclass
    class TestDocument(TextBasedDocument):
        spans: AnnotationLayer[LabeledSpan] = annotation_field(target="text")

    doc = TestDocument(text="Hello, world!")
    labeled_span = LabeledSpan(start=7, end=12, label="LOC")
    doc.spans.append(labeled_span)
    assert labeled_span.resolve() == ("LOC", "world")


def test_multilabeled_span():
    multilabeled_span1 = MultiLabeledSpan(start=1, end=2, label=("label1", "label2"))
    assert multilabeled_span1.start == 1
    assert multilabeled_span1.end == 2
    assert multilabeled_span1.label == ("label1", "label2")
    assert multilabeled_span1.score == pytest.approx((1.0, 1.0))

    multilabeled_span2 = MultiLabeledSpan(
        start=3, end=4, label=("label3", "label4"), score=(0.4, 0.5)
    )
    assert multilabeled_span2.start == 3
    assert multilabeled_span2.end == 4
    assert multilabeled_span2.label == ("label3", "label4")
    assert multilabeled_span2.score == pytest.approx((0.4, 0.5))

    assert multilabeled_span2.asdict() == {
        "_id": multilabeled_span2._id,
        "start": 3,
        "end": 4,
        "label": ("label3", "label4"),
        "score": (0.4, 0.5),
    }

    _test_annotation_reconstruction(multilabeled_span2)

    with pytest.raises(
        ValueError, match=re.escape("Number of labels (2) and scores (3) must be equal.")
    ):
        MultiLabeledSpan(start=5, end=6, label=("label5", "label6"), score=(0.1, 0.2, 0.3))

    with pytest.raises(ValueError) as excinfo:
        multilabeled_span1.resolve()
    assert (
        str(excinfo.value)
        == "MultiLabeledSpan(start=1, end=2, label=('label1', 'label2'), score=(1.0, 1.0)) is not attached to a target."
    )

    @dataclasses.dataclass
    class TestDocument(TextBasedDocument):
        spans: AnnotationLayer[MultiLabeledSpan] = annotation_field(target="text")

    doc = TestDocument(text="Hello, world!")
    multilabeled_span = MultiLabeledSpan(start=7, end=12, label=("LOC", "ORG"))
    doc.spans.append(multilabeled_span)
    assert multilabeled_span.resolve() == (("LOC", "ORG"), "world")


def test_labeled_multi_span():
    labeled_multi_span1 = LabeledMultiSpan(slices=((1, 2), (3, 4)), label="label1")
    assert labeled_multi_span1.slices == ((1, 2), (3, 4))
    assert labeled_multi_span1.label == "label1"
    assert labeled_multi_span1.score == pytest.approx(1.0)

    labeled_multi_span2 = LabeledMultiSpan(
        slices=((5, 6), (7, 8)),
        label="label2",
        score=0.5,
    )
    assert labeled_multi_span2.slices == ((5, 6), (7, 8))
    assert labeled_multi_span2.label == "label2"
    assert labeled_multi_span2.score == pytest.approx(0.5)

    assert labeled_multi_span2.asdict() == {
        "_id": labeled_multi_span2._id,
        "slices": ((5, 6), (7, 8)),
        "label": "label2",
        "score": 0.5,
    }

    _test_annotation_reconstruction(labeled_multi_span2)


def test_binary_relation():
    head = Span(start=1, end=2)
    tail = Span(start=3, end=4)

    binary_relation1 = BinaryRelation(head=head, tail=tail, label="label1")
    assert binary_relation1.head == head
    assert binary_relation1.tail == tail
    assert binary_relation1.label == "label1"
    assert binary_relation1.score == pytest.approx(1.0)

    binary_relation2 = BinaryRelation(head=head, tail=tail, label="label2", score=0.5)
    assert binary_relation2.head == head
    assert binary_relation2.tail == tail
    assert binary_relation2.label == "label2"
    assert binary_relation2.score == pytest.approx(0.5)

    assert binary_relation2.asdict() == {
        "_id": binary_relation2._id,
        "head": head._id,
        "tail": tail._id,
        "label": "label2",
        "score": 0.5,
    }

    annotation_store = {
        head._id: head,
        tail._id: tail,
    }
    _test_annotation_reconstruction(binary_relation2, annotation_store=annotation_store)

    with pytest.raises(
        ValueError,
        match=re.escape("Unable to resolve the annotation id without annotation_store."),
    ):
        BinaryRelation.fromdict(binary_relation2.asdict())

    with pytest.raises(ValueError) as excinfo:
        binary_relation1.resolve()
    assert str(excinfo.value) == "Span(start=1, end=2) is not attached to a target."

    @dataclasses.dataclass
    class TestDocument(TextBasedDocument):
        spans: AnnotationLayer[Span] = annotation_field(target="text")
        relations: AnnotationLayer[BinaryRelation] = annotation_field(target="spans")

    doc = TestDocument(text="Hello, world!")
    head = Span(start=0, end=5)
    tail = Span(start=7, end=12)
    doc.spans.extend([head, tail])
    relation = BinaryRelation(head=head, tail=tail, label="LABEL")
    doc.relations.append(relation)
    assert relation.resolve() == ("LABEL", ("Hello", "world"))


def test_multilabeled_binary_relation():
    head = Span(start=1, end=2)
    tail = Span(start=3, end=4)

    binary_relation1 = MultiLabeledBinaryRelation(head=head, tail=tail, label=("label1", "label2"))
    assert binary_relation1.head == head
    assert binary_relation1.tail == tail
    assert binary_relation1.label == ("label1", "label2")
    assert binary_relation1.score == pytest.approx((1.0, 1.0))

    binary_relation2 = MultiLabeledBinaryRelation(
        head=head, tail=tail, label=("label3", "label4"), score=(0.4, 0.5)
    )
    assert binary_relation2.head == head
    assert binary_relation2.tail == tail
    assert binary_relation2.label == ("label3", "label4")
    assert binary_relation2.score == pytest.approx((0.4, 0.5))

    assert binary_relation2.asdict() == {
        "_id": binary_relation2._id,
        "head": head._id,
        "tail": tail._id,
        "label": ("label3", "label4"),
        "score": (0.4, 0.5),
    }

    annotation_store = {
        head._id: head,
        tail._id: tail,
    }
    _test_annotation_reconstruction(binary_relation2, annotation_store=annotation_store)

    with pytest.raises(
        ValueError,
        match=re.escape("Unable to resolve the annotation id without annotation_store."),
    ):
        MultiLabeledBinaryRelation.fromdict(binary_relation2.asdict())

    with pytest.raises(
        ValueError, match=re.escape("Number of labels (2) and scores (3) must be equal.")
    ):
        MultiLabeledBinaryRelation(
            head=head, tail=tail, label=("label5", "label6"), score=(0.1, 0.2, 0.3)
        )

    with pytest.raises(ValueError) as excinfo:
        binary_relation1.resolve()
    assert str(excinfo.value) == "Span(start=1, end=2) is not attached to a target."

    @dataclasses.dataclass
    class TestDocument(TextBasedDocument):
        spans: AnnotationLayer[Span] = annotation_field(target="text")
        relations: AnnotationLayer[MultiLabeledBinaryRelation] = annotation_field(target="spans")

    doc = TestDocument(text="Hello, world!")
    head = Span(start=0, end=5)
    tail = Span(start=7, end=12)
    doc.spans.extend([head, tail])
    relation = MultiLabeledBinaryRelation(head=head, tail=tail, label=("LABEL1", "LABEL2"))
    doc.relations.append(relation)
    assert relation.resolve() == (("LABEL1", "LABEL2"), ("Hello", "world"))


def test_nary_relation():
    arg1 = Span(start=1, end=2)
    arg2 = Span(start=3, end=4)
    arg3 = Span(start=5, end=6)

    nary_relation1 = NaryRelation(
        arguments=(arg1, arg2, arg3), roles=("role1", "role2", "role3"), label="label1"
    )

    assert nary_relation1.arguments == (arg1, arg2, arg3)
    assert nary_relation1.roles == ("role1", "role2", "role3")
    assert nary_relation1.label == "label1"
    assert nary_relation1.score == pytest.approx(1.0)

    assert nary_relation1.asdict() == {
        "_id": nary_relation1._id,
        "arguments": [arg1._id, arg2._id, arg3._id],
        "roles": ("role1", "role2", "role3"),
        "label": "label1",
        "score": 1.0,
    }

    annotation_store = {
        arg1._id: arg1,
        arg2._id: arg2,
        arg3._id: arg3,
    }
    _test_annotation_reconstruction(nary_relation1, annotation_store=annotation_store)

    with pytest.raises(
        ValueError,
        match=re.escape("Unable to resolve the annotation id without annotation_store."),
    ):
        NaryRelation.fromdict(nary_relation1.asdict())

    with pytest.raises(ValueError) as excinfo:
        nary_relation1.resolve()
    assert str(excinfo.value) == "Span(start=1, end=2) is not attached to a target."

    @dataclasses.dataclass
    class TestDocument(TextBasedDocument):
        spans: AnnotationLayer[Span] = annotation_field(target="text")
        relations: AnnotationLayer[NaryRelation] = annotation_field(target="spans")

    doc = TestDocument(text="Hello, world A and B!")
    arg1 = Span(start=0, end=5)
    arg2 = Span(start=7, end=14)
    arg3 = Span(start=19, end=20)
    doc.spans.extend([arg1, arg2, arg3])
    relation = NaryRelation(
        arguments=(arg1, arg2, arg3), roles=("ARG1", "ARG2", "ARG3"), label="LABEL"
    )
    doc.relations.append(relation)
    assert relation.resolve() == (
        "LABEL",
        (("ARG1", "Hello"), ("ARG2", "world A"), ("ARG3", "B")),
    )
