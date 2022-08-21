import dataclasses
import re

import pytest

from pytorch_ie.annotations import BinaryRelation, Label, LabeledSpan, Span
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.core.document import _enumerate_dependencies
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
        label: AnnotationList[Label] = annotation_field()

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
    assert set(document1._annotation_graph.keys()) == {
        "sentences",
        "relations",
        "entities",
        "_artificial_root",
    }
    assert set(document1._annotation_graph["sentences"]) == {"text"}
    assert set(document1._annotation_graph["relations"]) == {"entities"}
    assert set(document1._annotation_graph["entities"]) == {"text"}
    assert set(document1._annotation_graph["_artificial_root"]) == {
        "sentences",
        "relations",
        "label",
    }

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

    assert len(document1) == 4
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
    # add a prediction that is also an annotation
    document1.entities.predictions.append(labeled_span1)

    assert len(document1.sentences.predictions) == 2
    assert document1.sentences.predictions[1].target == document1.text
    assert len(document1["sentences"].predictions) == 2
    assert document1["sentences"].predictions[1].target == document1.text

    document1.label.append(Label(label="test_label", score=1.0))

    assert document1 == TestDocument.fromdict(document1.asdict())

    # number of annotation fields
    assert len(document1) == 4
    # actual annotation fields (tests __iter__)
    assert set(document1) == {"sentences", "entities", "relations", "label"}


def test_as_type():
    @dataclasses.dataclass
    class TestDocument1(TextDocument):
        sentences: AnnotationList[Span] = annotation_field(target="text")
        entities: AnnotationList[LabeledSpan] = annotation_field(target="text")

    @dataclasses.dataclass
    class TestDocument2(TextDocument):
        sentences: AnnotationList[Span] = annotation_field(target="text")
        ents: AnnotationList[LabeledSpan] = annotation_field(target="text")

    @dataclasses.dataclass
    class TestDocument3(TextDocument):
        entities: AnnotationList[LabeledSpan] = annotation_field(target="text")
        relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")

    # create input document with "sentences" and "relations"
    document1 = TestDocument1(text="test1")
    span1 = Span(start=1, end=2)
    span2 = Span(start=3, end=4)
    document1.sentences.append(span1)
    document1.sentences.append(span2)
    labeled_span1 = LabeledSpan(start=1, end=2, label="label1")
    labeled_span2 = LabeledSpan(start=3, end=4, label="label2")
    document1.entities.append(labeled_span1)
    document1.entities.append(labeled_span2)

    # convert rename "entities" to "ents"
    document2 = document1.as_type(new_type=TestDocument2, field_mapping={"entities": "ents"})
    assert set(document2) == {"sentences", "ents"}
    assert document2.sentences == document1.sentences
    assert document2.ents == document1.entities

    # remove "sentences", but add "relations"
    document3 = document1.as_type(new_type=TestDocument3)
    assert set(document3) == {"entities", "relations"}
    rel = BinaryRelation(head=span1, tail=span2, label="rel")
    document3.relations.append(rel)
    assert len(document3.relations) == 1


def test_enumerate_dependencies():
    # annotation field -> targets
    graph = {"a": ["b"], "b": ["c"], "d": ["c", "a"], "e": ["f"], "g": ["e"], "h": ["e"]}
    root_nodes = ["d", "g", "h"]
    resolved = []
    _enumerate_dependencies(resolved=resolved, dependency_graph=graph, nodes=root_nodes)

    for i, node in enumerate(resolved):
        already_resolved = resolved[:i]
        targets = graph.get(node, [])
        for t in targets:
            assert t in already_resolved


def test_enumerate_dependencies_with_circle():
    graph = {"a": ["b"], "b": ["c"], "c": ["b"], "d": ["e"]}
    root_nodes = ["a", "d"]
    resolved = []
    with pytest.raises(ValueError, match=re.escape("circular dependency detected at node: b")):
        _enumerate_dependencies(resolved=resolved, dependency_graph=graph, nodes=root_nodes)
