import re
from dataclasses import dataclass

import pytest

from pytorch_ie.annotations import (
    BinaryRelation,
    Label,
    LabeledMultiSpan,
    LabeledSpan,
    MultiLabel,
    MultiLabeledBinaryRelation,
    MultiLabeledMultiSpan,
    MultiLabeledSpan,
    Span,
)
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.documents import TextDocument


def test_label():
    label1 = Label(label="label1")
    assert label1.label == "label1"
    assert label1.score == pytest.approx(1.0)

    label2 = Label(label="label2", score=0.5)
    assert label2.label == "label2"
    assert label2.score == pytest.approx(0.5)

    assert label2.asdict() == {
        "_id": hash(label2),
        "label": "label2",
        "score": 0.5,
    }

    assert label2 == Label.fromdict(label2.asdict())


def test_multilabel():
    multilabel1 = MultiLabel(label=("label1", "label2"))
    assert multilabel1.label == ("label1", "label2")
    assert multilabel1.score == pytest.approx((1.0, 1.0))

    multilabel2 = MultiLabel(label=("label3", "label4"), score=(0.4, 0.5))
    assert multilabel2.label == ("label3", "label4")
    assert multilabel2.score == pytest.approx((0.4, 0.5))

    multilabel2.asdict() == {
        "_id": hash(multilabel2),
        "label": ("label3", "label4"),
        "score": (0.4, 0.5),
    }

    assert multilabel2 == MultiLabel.fromdict(multilabel2.asdict())

    with pytest.raises(
        ValueError, match=re.escape("Number of labels (2) and scores (3) must be equal.")
    ):
        MultiLabel(label=["label5", "label6"], score=[0.1, 0.2, 0.3])


def test_span():
    span = Span(start=1, end=2)
    assert span.start == 1
    assert span.end == 2

    span.asdict() == {
        "_id": hash(span),
        "start": 1,
        "end": 2,
    }

    assert span == Span.fromdict(span.asdict())


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

    labeled_span2.asdict() == {
        "_id": hash(labeled_span2),
        "label": ("label3", "label4"),
        "score": (0.4, 0.5),
    }

    assert labeled_span2 == LabeledSpan.fromdict(labeled_span2.asdict())


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

    multilabeled_span2.asdict() == {
        "_id": hash(multilabeled_span2),
        "start": 3,
        "end": 4,
        "label": ("label3", "label4"),
        "score": (0.4, 0.5),
    }

    assert multilabeled_span2 == MultiLabeledSpan.fromdict(multilabeled_span2.asdict())

    with pytest.raises(
        ValueError, match=re.escape("Number of labels (2) and scores (3) must be equal.")
    ):
        MultiLabeledSpan(start=5, end=6, label=["label5", "label6"], score=[0.1, 0.2, 0.3])


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

    labeled_multi_span2.asdict() == {
        "_id": hash(labeled_multi_span2),
        "slices": ((5, 6), (7, 8)),
        "label": "label2",
        "score": 0.5,
    }

    assert labeled_multi_span2 == LabeledMultiSpan.fromdict(labeled_multi_span2.asdict())


def test_multilabeled_multi_span():
    multilabeled_multi_span1 = MultiLabeledMultiSpan(
        slices=((1, 2), (3, 4)), label=("label1", "label2")
    )
    assert multilabeled_multi_span1.slices == ((1, 2), (3, 4))
    assert multilabeled_multi_span1.label == ("label1", "label2")
    assert multilabeled_multi_span1.score == pytest.approx((1.0, 1.0))

    multilabeled_multi_span2 = MultiLabeledMultiSpan(
        slices=((5, 6), (7, 8)), label=("label3", "label4"), score=(0.4, 0.5)
    )
    assert multilabeled_multi_span2.slices == ((5, 6), (7, 8))
    assert multilabeled_multi_span2.label == ("label3", "label4")
    assert multilabeled_multi_span2.score == pytest.approx((0.4, 0.5))

    multilabeled_multi_span2.asdict() == {
        "_id": hash(multilabeled_multi_span2),
        "slices": ((5, 6), (7, 8)),
        "label": ("label2", "label3"),
        "score": (0.4, 0.5),
    }

    assert multilabeled_multi_span2 == MultiLabeledMultiSpan.fromdict(
        multilabeled_multi_span2.asdict()
    )

    with pytest.raises(
        ValueError, match=re.escape("Number of labels (2) and scores (3) must be equal.")
    ):
        MultiLabeledMultiSpan(
            slices=((9, 10), (11, 12)), label=("label5", "label6"), score=(0.1, 0.2, 0.3)
        )


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

    binary_relation2.asdict() == {
        "_id": hash(binary_relation2),
        "head": hash(head),
        "tail": hash(tail),
        "label": "label2",
        "score": 0.5,
    }

    annotation_store = {
        hash(head): ("head", head),
        hash(tail): ("tail", tail),
    }
    binary_relation2 == BinaryRelation.fromdict(
        binary_relation2.asdict(), annotation_store=annotation_store
    )

    with pytest.raises(
        ValueError, match=re.escape("Unable to resolve head reference without annotation_store.")
    ):
        binary_relation2 == BinaryRelation.fromdict(binary_relation2.asdict())


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

    binary_relation2.asdict() == {
        "_id": hash(binary_relation2),
        "head": hash(head),
        "tail": hash(tail),
        "label": ("label3", "label4"),
        "score": (0.4, 0.5),
    }

    annotation_store = {
        hash(head): ("head", head),
        hash(tail): ("tail", tail),
    }
    binary_relation2 == MultiLabeledBinaryRelation.fromdict(
        binary_relation2.asdict(), annotation_store=annotation_store
    )

    with pytest.raises(
        ValueError, match=re.escape("Unable to resolve head reference without annotation_store.")
    ):
        binary_relation2 == MultiLabeledBinaryRelation.fromdict(binary_relation2.asdict())

    with pytest.raises(
        ValueError, match=re.escape("Number of labels (2) and scores (3) must be equal.")
    ):
        MultiLabeledBinaryRelation(
            head=head, tail=tail, label=("label5", "label6"), score=(0.1, 0.2, 0.3)
        )
