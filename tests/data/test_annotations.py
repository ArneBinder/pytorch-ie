import re
from dataclasses import dataclass

import pytest

from pytorch_ie.data import (
    AnnotationList,
    Label,
    LabeledSpan,
    MultiLabel,
    MultiLabeledSpan,
    TextDocument,
)
from pytorch_ie.data.document import annotation_field


def test_label():
    label1 = Label(label="label1")
    assert label1.label == "label1"
    assert label1.score == pytest.approx(1.0)

    label2 = Label(label="label2", score=0.5)
    assert label2.label == "label2"
    assert label2.score == pytest.approx(0.5)

    assert label2.asdict() == {
        "id": hash(label2),
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
        "id": hash(multilabel2),
        "label": ("label3", "label4"),
        "score": (0.4, 0.5),
    }

    assert multilabel2 == MultiLabel.fromdict(multilabel2.asdict())

    with pytest.raises(
        ValueError, match=re.escape("Number of labels (2) and scores (3) must be equal.")
    ):
        MultiLabel(label=["label5", "label6"], score=[0.1, 0.2, 0.3])


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
        "id": hash(labeled_span2),
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
        "id": hash(multilabeled_span2),
        "start": 3,
        "end": 4,
        "label": ("label3", "label4"),
        "score": (0.4, 0.5),
    }

    assert multilabeled_span2 == MultiLabeledSpan.fromdict(multilabeled_span2.asdict())


def test_annotation_list():
    @dataclass
    class TestDocument(TextDocument):
        entities: AnnotationList[LabeledSpan] = annotation_field(target="text")

    document = TestDocument(text="Entity A works at B.")

    entity1 = LabeledSpan(start=0, end=8, label="PER")
    entity2 = LabeledSpan(start=18, end=19, label="ORG")

    document.entities.append(entity1)
    document.entities.append(entity2)

    entity3 = LabeledSpan(start=18, end=19, label="PRED-ORG")
    entity4 = LabeledSpan(start=0, end=8, label="PRED-PER")

    document.entities.predictions.append(entity3)
    document.entities.predictions.append(entity4)

    assert isinstance(document.entities, AnnotationList)
    assert len(document.entities) == 2
    assert document.entities[0] == entity1
    assert document.entities[1] == entity2
    assert document.entities[0].target == document.text
    assert document.entities[1].target == document.text
    assert document.entities[0].text == "Entity A"
    assert document.entities[1].text == "B"

    assert len(document.entities.predictions) == 2
    assert document.entities.predictions[0] == entity3
    assert document.entities.predictions[1] == entity4
    assert document.entities.predictions[0].target == document.text
    assert document.entities.predictions[1].target == document.text
    assert document.entities.predictions[0].text == "B"
    assert document.entities.predictions[1].text == "Entity A"

    document.entities.clear()
    assert len(document.entities) == 0

    document.entities.predictions.clear()
    assert len(document.entities.predictions) == 0
