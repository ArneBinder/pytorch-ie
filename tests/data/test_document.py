import pytest

from pytorch_ie.data import (
    Annotation,
    BinaryRelation,
    Document,
    Label,
    LabeledMultiSpan,
    LabeledSpan,
)


def test_labeled_multi_span():
    labeled_multi_span1 = LabeledMultiSpan(slices=[(1, 2), (3, 4)], label="test", score=1.0)
    assert (
        str(labeled_multi_span1)
        == "LabeledMultiSpan(slices=[(1, 2), (3, 4)], label=test, score=1.0, metadata={})"
    )

    labeled_multi_span2 = LabeledMultiSpan(
        slices=[(1, 2), (3, 4)], label=["test", "test2"], score=[1.0, 2.0]
    )
    assert (
        str(labeled_multi_span2)
        == "LabeledMultiSpan(slices=[(1, 2), (3, 4)], label=['test', 'test2'], score=[1.0, 2.0], metadata={})"
    )

    labeled_multi_span3 = LabeledMultiSpan.from_dict(
        dict(slices=[(1, 2), (3, 4)], label="test", score=1.0, metadata={"test": "test"})
    )
    assert labeled_multi_span3.slices == [(1, 2), (3, 4)]
    assert labeled_multi_span3.label == "test"
    assert labeled_multi_span3.score == 1.0
    assert labeled_multi_span3.metadata == {"test": "test"}


def test_binary_relation():
    head = LabeledSpan(start=1, end=2, label="head", score=1.0)
    tail = LabeledSpan(start=3, end=4, label="tail", score=1.0)

    binary_relation1 = BinaryRelation(head=head, tail=tail, label="test", score=1.0)
    assert (
        str(binary_relation1)
        == "BinaryRelation(head=LabeledSpan(start=1, end=2, label=head, score=1.0, metadata={}), tail=LabeledSpan(start=3, end=4, label=tail, score=1.0, metadata={}), label=test, score=1.0, metadata={})"
    )


def test_document():
    document = Document(text="test", doc_id="id")

    document.metadata["test"] = "test"
    document.add_annotation("annotation", Label(label="annotation_label", score=1.0))
    document.add_prediction("prediction", Label(label="prediction_label", score=1.0))

    assert document.text == "test"
    assert document.id == "id"
    assert document.metadata == {"test": "test"}

    annotation = document.annotations.labels["annotation"]
    assert len(annotation) == 1
    assert annotation[0].label == "annotation_label"

    prediction1 = document.predictions.labels["prediction"]
    assert len(prediction1) == 1
    assert prediction1[0].label == "prediction_label"

    document.clear_predictions("prediction")
    assert not document.predictions.labels.has_layer("prediction")
