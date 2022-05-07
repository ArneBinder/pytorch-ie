import dataclasses
import re

import pytest

from pytorch_ie.annotations import BinaryRelation, LabeledSpan, Span
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.documents import TextDocument


def test_text_document():
    document1 = TextDocument(text="text1")
    assert document1.text == "text1"
    assert document1.id is None
    assert document1.metadata == {}

    document1.asdict() == {
        "id": None,
        "text": "text1",
    }

    assert document1 == TextDocument.fromdict(document1.asdict())

    document2 = TextDocument(text="text2", id="test_id", metadata={"key": "value"})
    assert document2.text == "text2"
    assert document2.id == "test_id"
    assert document2.metadata == {"key": "value"}

    document2.asdict() == {
        "id": "test_id",
        "text": "text1",
        "metadata": {
            "key": "value",
        },
    }

    assert document2 == TextDocument.fromdict(document2.asdict())


def test_document_with_annotations():
    @dataclasses.dataclass
    class TestDocument(TextDocument):
        sentences: AnnotationList[Span] = annotation_field(target="text")
        entities: AnnotationList[LabeledSpan] = annotation_field(target="text")
        relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")

    document1 = TestDocument(text="test1")
    assert isinstance(document1.sentences, AnnotationList)
    assert isinstance(document1.entities, AnnotationList)
    assert isinstance(document1.relations, AnnotationList)
    assert len(document1.sentences) == 0
    assert len(document1.entities) == 0
    assert len(document1.relations) == 0
    assert len(document1.sentences.predictions) == 0
    assert len(document1.entities.predictions) == 0
    assert len(document1.relations.predictions) == 0
    assert set(document1._annotation_graph.keys()) == {"text", "entities"}
    assert set(document1._annotation_graph["text"]) == {"sentences", "entities"}
    assert set(document1._annotation_graph["entities"]) == {"relations"}

    span1 = Span(start=1, end=2)
    span2 = Span(start=3, end=4)

    document1.sentences.append(span1)
    document1.sentences.append(span2)
    assert len(document1.sentences) == 2
    assert document1.sentences[:2] == [span1, span2]
    assert document1.sentences[0].target == document1.text

    labeled_span1 = LabeledSpan(start=1, end=2, label="label1")
    labeled_span2 = LabeledSpan(start=3, end=4, label="label2")
    document1.entities.append(labeled_span1)
    document1.entities.append(labeled_span2)
    assert len(document1.entities) == 2
    assert document1.sentences[0].target == document1.text

    relation1 = BinaryRelation(head=labeled_span1, tail=labeled_span2, label="label1")

    document1.relations.append(relation1)
    assert len(document1.relations) == 1
    assert document1.relations[0].target == document1.entities

    assert document1 == TestDocument.fromdict(document1.asdict())

    assert len(document1) == 3
    assert len(document1["sentences"]) == 2
    assert document1["sentences"][0].target == document1.text

    with pytest.raises(
        KeyError, match=re.escape("Document has no attribute 'non_existing_annotation'.")
    ):
        document1["non_existing_annotation"]

    span3 = Span(start=5, end=6)
    span4 = Span(start=7, end=8)

    document1.sentences.predictions.append(span3)
    document1.sentences.predictions.append(span4)
    assert len(document1.sentences.predictions) == 2
    assert document1.sentences.predictions[1].target == document1.text
    assert len(document1["sentences"].predictions) == 2
    assert document1["sentences"].predictions[1].target == document1.text

    # TODO: revisit when we decided how to handle serialization of predictions
    # assert document1 == TestDocument.fromdict(document1.asdict())


@pytest.mark.parametrize("overwrite", [False, True])
def test_integrate_annotations(overwrite):
    @dataclasses.dataclass
    class TestDocument(TextDocument):
        entities: AnnotationList[LabeledSpan] = annotation_field(target="text")

    document = TestDocument(text="Entity A works at B.")

    entity1 = LabeledSpan(start=0, end=8, label="PER")
    entity2 = LabeledSpan(start=18, end=19, label="ORG")

    document.entities.append(entity1)
    document.entities.predictions.append(entity2)

    assert len(document.entities) == 1
    assert document.entities[0] == entity1

    assert len(document.entities.predictions) == 1
    assert document.entities.predictions[0] == entity2

    document.entities.integrate_predictions(overwrite=overwrite)

    if overwrite:
        assert len(document.entities) == 1
        assert document.entities[0] == entity2
    else:
        assert len(document.entities) == 2
        assert document.entities[0] == entity1
        assert document.entities[1] == entity2

    assert len(document.entities.predictions) == 0
