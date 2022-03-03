import pytest

from pytorch_ie.data.document import (
    Annotation,
    AnnotationLayer,
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


def test_annotation_layer_add():
    label1 = Label("label1")
    label2 = Label("label2")
    layer = AnnotationLayer()
    layer.append(label1)
    layer.append(label2)
    assert len(layer) == 2


def test_annotation_layer_get_labels():
    label1 = Label("label1")
    label2 = Label("label2")
    layer = AnnotationLayer()
    layer.extend([label1, label2])
    labels = layer.as_labels
    assert len(labels) == 2
    with pytest.raises(TypeError) as e:
        spans = layer.as_spans
    assert (
        str(e.value)
        == "Entry caused a type mismatch. Expected type: <class 'pytorch_ie.data.document.Label'>, actual type: <class 'pytorch_ie.data.document.LabeledSpan'>."
    )
    with pytest.raises(TypeError) as e:
        rels = layer.as_binary_relations
    assert (
        str(e.value)
        == "Entry caused a type mismatch. Expected type: <class 'pytorch_ie.data.document.Label'>, actual type: <class 'pytorch_ie.data.document.BinaryRelation'>."
    )


def test_annotation_layer_get_spans():
    span1 = LabeledSpan(label="span1", start=0, end=4)
    span2 = LabeledSpan(label="span2", start=5, end=6)
    layer = AnnotationLayer()
    layer.extend([span1, span2])
    spans = layer.as_spans
    assert len(spans) == 2
    with pytest.raises(TypeError) as e:
        labels = layer.as_labels
    assert (
        str(e.value)
        == "Entry caused a type mismatch. Expected type: <class 'pytorch_ie.data.document.LabeledSpan'>, actual type: <class 'pytorch_ie.data.document.Label'>."
    )
    with pytest.raises(TypeError) as e:
        rels = layer.as_binary_relations
    assert (
        str(e.value)
        == "Entry caused a type mismatch. Expected type: <class 'pytorch_ie.data.document.LabeledSpan'>, actual type: <class 'pytorch_ie.data.document.BinaryRelation'>."
    )


def test_annotation_layer_get_rels():
    span1 = LabeledSpan(label="span1", start=0, end=4)
    span2 = LabeledSpan(label="span2", start=5, end=6)
    rel1 = BinaryRelation(label="rel1", head=span1, tail=span2)
    rel2 = BinaryRelation(label="rel2", head=span2, tail=span1)
    layer = AnnotationLayer()
    layer.extend([rel1, rel2])
    rels = layer.as_binary_relations
    assert len(rels) == 2
    with pytest.raises(TypeError) as e:
        labels = layer.as_labels
    assert (
        str(e.value)
        == "Entry caused a type mismatch. Expected type: <class 'pytorch_ie.data.document.BinaryRelation'>, actual type: <class 'pytorch_ie.data.document.Label'>."
    )
    with pytest.raises(TypeError) as e:
        spans = layer.as_spans
    assert (
        str(e.value)
        == "Entry caused a type mismatch. Expected type: <class 'pytorch_ie.data.document.BinaryRelation'>, actual type: <class 'pytorch_ie.data.document.LabeledSpan'>."
    )


def test_annotation_layer_add_wrong_type():
    layer = AnnotationLayer()
    label = Label("label1")
    span = LabeledSpan(start=0, end=2, label="span")
    layer.append(label)
    with pytest.raises(TypeError) as e:
        layer.append(span)
    assert (
        str(e.value)
        == "Entry caused a type mismatch. Expected type: <class 'pytorch_ie.data.document.Label'>, actual type: <class 'pytorch_ie.data.document.LabeledSpan'>."
    )


def test_annotation_layer_extend_wrong_type():
    layer = AnnotationLayer()
    label = Label("label1")
    span = LabeledSpan(start=0, end=2, label="span")
    layer.append(label)
    with pytest.raises(TypeError) as e:
        layer.extend([span])
    assert (
        str(e.value)
        == "Entry caused a type mismatch. Expected type: <class 'pytorch_ie.data.document.Label'>, actual type: <class 'pytorch_ie.data.document.LabeledSpan'>."
    )


def test_annotation_layer_set_wrong_type():
    layer = AnnotationLayer()
    label = Label("label1")
    span = LabeledSpan(start=0, end=2, label="span")
    layer.append(label)
    with pytest.raises(TypeError) as e:
        layer[0] = span
    assert (
        str(e.value)
        == "Entry caused a type mismatch. Expected type: <class 'pytorch_ie.data.document.Label'>, actual type: <class 'pytorch_ie.data.document.LabeledSpan'>."
    )
