import pytest

from pytorch_ie.data.document import (
    Annotation,
    BinaryRelation,
    Document,
    Label,
    LabeledMultiSpan,
    LabeledSpan,
)


def test_annotation():
    annotation1 = Annotation(label="label", score=0.5, metadata={"test": "test"})
    assert annotation1.label == "label"
    assert annotation1.score == 0.5
    assert annotation1.metadata == {"test": "test"}

    annotation2 = Annotation.from_dict(dict(label="label", score=0.5, metadata={"test": "test"}))
    assert annotation2.label == "label"
    assert annotation2.score == 0.5
    assert annotation2.metadata == {"test": "test"}


def test_annotation_no_score():
    annotation = Annotation(label="label")
    assert annotation.label == "label"
    assert annotation.score == 1.0


def test_annotation_multilabel():
    annotation = Annotation(label=["label1", "label2"])
    assert annotation.is_multilabel
    assert annotation.label == ["label1", "label2"]
    assert annotation.score == [1.0, 1.0]


def test_annotation_incorrect():
    with pytest.raises(ValueError, match="Too many scores for label."):
        Annotation(label="label", score=[1.0, 1.0])

    with pytest.raises(ValueError, match="Multi-label requires score to be a list."):
        Annotation(label=["label1", "label2"], score=1.0)

    with pytest.raises(ValueError, match="Number of labels and scores must be equal."):
        Annotation(label=["label1", "label2"], score=[1.0, 1.0, 1.0])


def test_label():
    label1 = Label(label="test", score=1.0)
    assert str(label1) == "Label(label=test, score=1.0)"

    label2 = Label(label=["test", "test2"], score=[1.0, 2.0])
    assert str(label2) == "Label(label=['test', 'test2'], score=[1.0, 2.0])"

    label3 = Label.from_dict(dict(label="test", score=1.0, metadata={"test": "test"}))
    assert label3.label == "test"
    assert label3.score == 1.0
    assert label3.metadata == {"test": "test"}


def test_labeled_span():
    labeled_span1 = LabeledSpan(start=1, end=2, label="test", score=1.0)
    assert str(labeled_span1) == "LabeledSpan(start=1, end=2, label=test, score=1.0, metadata={})"

    labeled_span2 = LabeledSpan(start=1, end=2, label=["test", "test2"], score=[1.0, 2.0])
    assert (
        str(labeled_span2)
        == "LabeledSpan(start=1, end=2, label=['test', 'test2'], score=[1.0, 2.0], metadata={})"
    )

    labeled_span3 = LabeledSpan.from_dict(
        dict(start=1, end=2, label="test", score=1.0, metadata={"test": "test"})
    )
    assert labeled_span3.start == 1
    assert labeled_span3.end == 2
    assert labeled_span3.label == "test"
    assert labeled_span3.score == 1.0
    assert labeled_span3.metadata == {"test": "test"}


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

    annotation = document.annotations["annotation"].as_labels
    assert len(annotation) == 1
    assert annotation[0].label == "annotation_label"

    prediction1 = document.predictions["prediction"].as_labels
    assert len(prediction1) == 1
    assert prediction1[0].label == "prediction_label"

    document.clear_predictions("prediction")
    assert not document.predictions.has_layer("prediction")


# def test_layer_cast():
# document = Document(text="test", doc_id="id")
# document.add_annotation(
#    "annotation", LabeledSpan(start=1, end=3, label="annotation_label", score=1.0)
# )

# span_layer = document.annotations["annotation"].cast()
# print(type(span_layer))
