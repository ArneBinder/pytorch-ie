import dataclasses
import re

import pytest

from pytorch_ie.annotations import BinaryRelation, Label, LabeledSpan, Span
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.core.document import Annotation, Document, _enumerate_dependencies
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
    relation2 = BinaryRelation(head=labeled_span1, tail=labeled_span2, label="label1")
    relation3 = BinaryRelation(head=labeled_span2, tail=labeled_span1, label="label1")
    assert relation1.id == relation2.id
    assert relation1.id != relation3.id

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
    # remove the annotation to allow reassigning it
    relation1_popped = document1.relations.pop(0)
    assert relation1_popped == relation1
    document1.relations.predictions.append(relation1)

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


def test_document_with_same_annotations():
    @dataclasses.dataclass
    class TestDocument(Document):
        text: str
        text2: str
        text3: str
        tokens0: AnnotationList[Span] = annotation_field(target="text")
        tokens1: AnnotationList[Span] = annotation_field(target="text")
        tokens2: AnnotationList[Span] = annotation_field(target="text2")
        tokens3: AnnotationList[Span] = annotation_field(target="text3")

    doc = TestDocument(text="test1", text2="test1", text3="test2")
    start = 0
    end = len(doc.text)
    token0 = Span(start=start, end=end)
    token1 = Span(start=start, end=end)
    token2 = Span(start=start, end=end)
    token3 = Span(start=start, end=end)
    token0_id = token0.id
    token1_id = token1.id
    token2_id = token2.id
    token3_id = token3.id
    # all spans are identical, so are there ids
    assert token1_id == token0_id
    assert token2_id == token0_id
    assert token3_id == token0_id
    doc.tokens0.append(token0)
    doc.tokens1.append(token1)
    doc.tokens2.append(token2)
    doc.tokens3.append(token3)
    token0_id_added = token0.id
    token1_id_added = token1.id
    token2_id_added = token2.id
    token3_id_added = token3.id
    # after adding them to a document, the targets are taken into account, so their id changed
    assert token0_id_added != token0_id
    assert token1_id_added != token1_id
    assert token2_id_added != token2_id
    assert token3_id_added != token3_id

    # token0 and token1 spans are still identical because they have the same target
    assert token0_id_added == token1_id_added
    # they are also still identical with token2 because they have the same target content (!)
    assert token2_id_added == token0_id_added
    # teh differ from token3 because their target content (!) is different
    assert token3_id_added != token0_id_added

    # test reconstruction
    doc_dict = doc.asdict()
    doc_reconstructed = TestDocument.fromdict(doc_dict)
    assert doc == doc_reconstructed


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


def test_annotation_list_wrong_target():
    @dataclasses.dataclass
    class TestDocument(TextDocument):
        entities: AnnotationList[LabeledSpan] = annotation_field(target="does_not_exist")

    with pytest.raises(
        TypeError,
        match=re.escape(
            'annotation target "does_not_exist" is not in field names of the document: '
        ),
    ):
        document = TestDocument(text="text")


def test_annotation_list():
    @dataclasses.dataclass
    class TestDocument(TextDocument):
        entities: AnnotationList[LabeledSpan] = annotation_field(target="text")

    document = TestDocument(text="Entity A works at B.")

    entity1 = LabeledSpan(start=0, end=8, label="PER")
    entity2 = LabeledSpan(start=18, end=19, label="ORG")
    assert entity1.target is None
    assert entity2.target is None

    document.entities.append(entity1)
    document.entities.append(entity2)

    entity3 = LabeledSpan(start=18, end=19, label="PRED-ORG")
    entity4 = LabeledSpan(start=0, end=8, label="PRED-PER")
    assert entity3.target is None
    assert entity4.target is None

    document.entities.predictions.append(entity3)
    document.entities.predictions.append(entity4)

    assert isinstance(document.entities, AnnotationList)
    assert len(document.entities) == 2
    assert document.entities[0] == entity1
    assert document.entities[1] == entity2
    assert document.entities[0].target == document.text
    assert document.entities[1].target == document.text
    assert entity1.target == document.text
    assert entity2.target == document.text
    assert str(document.entities[0]) == "Entity A"
    assert str(document.entities[1]) == "B"

    assert len(document.entities.predictions) == 2
    assert document.entities.predictions[0] == entity3
    assert document.entities.predictions[1] == entity4
    assert document.entities.predictions[0].target == document.text
    assert document.entities.predictions[1].target == document.text
    assert entity3.target == document.text
    assert entity4.target == document.text
    assert str(document.entities.predictions[0]) == "B"
    assert str(document.entities.predictions[1]) == "Entity A"

    document.entities.clear()
    assert len(document.entities) == 0
    assert entity1.target is None
    assert entity2.target is None

    document.entities.predictions.clear()
    assert len(document.entities.predictions) == 0
    assert entity3.target is None
    assert entity4.target is None


def test_annotation_list_with_multiple_targets():
    @dataclasses.dataclass
    class TestDocument(TextDocument):
        entities1: AnnotationList[LabeledSpan] = annotation_field(target="text")
        entities2: AnnotationList[LabeledSpan] = annotation_field(target="text")
        relations: AnnotationList[BinaryRelation] = annotation_field(
            targets=["entities1", "entities2"]
        )
        label: AnnotationList[Label] = annotation_field()

    doc = TestDocument(text="test1")

    assert set(doc._annotation_graph.keys()) == {
        "entities1",
        "entities2",
        "relations",
        "_artificial_root",
    }
    assert set(doc._annotation_graph["entities1"]) == {"text"}
    assert set(doc._annotation_graph["entities2"]) == {"text"}
    assert set(doc._annotation_graph["relations"]) == {"entities1", "entities2"}
    assert set(doc._annotation_graph["_artificial_root"]) == {
        "relations",
        "label",
    }

    span1 = LabeledSpan(0, 2, label="a")
    assert span1.targets is None
    doc.entities1.append(span1)
    assert doc.entities1[0] == span1
    assert span1.target == doc.text

    span2 = LabeledSpan(2, 4, label="b")
    assert span2.targets is None
    doc.entities2.append(span2)
    assert doc.entities2[0] == span2
    assert span2.target == doc.text

    relation = BinaryRelation(head=span1, tail=span2, label="relation")
    assert relation.targets is None
    doc.relations.append(relation)
    assert doc.relations[0] == relation
    with pytest.raises(
        ValueError,
        match=re.escape("annotation has multiple targets, target is not defined in this case"),
    ):
        relation.target
    assert relation.targets == (doc.entities1, doc.entities2)

    label = Label("label")
    assert label.target is None
    doc.label.append(label)
    assert doc.label[0] == label
    with pytest.raises(ValueError, match=re.escape("annotation has no target")):
        label.target
    assert label.targets == ()


@dataclasses.dataclass(eq=True, frozen=True)
class DoubleTextSpan(Annotation):
    TARGET_NAMES = (
        "text1",
        "text2",
    )
    start1: int
    end1: int
    start2: int
    end2: int

    def __str__(self) -> str:
        if self.targets is None:
            return ""
        text1: str = self.named_targets["text1"]  # type: ignore
        text2: str = self.named_targets["text2"]  # type: ignore
        return str(text1[self.start1 : self.end1]) + "|" + str(text2[self.start2 : self.end2])


def test_annotation_list_with_named_targets():
    @dataclasses.dataclass
    class TestDocument(Document):
        texta: str
        textb: str
        entities1: AnnotationList[LabeledSpan] = annotation_field(target="texta")
        entities2: AnnotationList[LabeledSpan] = annotation_field(target="textb")
        # note that the entries in targets do not follow the order of DoubleTextSpan.TARGET_NAMES
        crossrefs: AnnotationList[DoubleTextSpan] = annotation_field(
            named_targets={"text2": "textb", "text1": "texta"}
        )

    doc = TestDocument(texta="text1", textb="text2")

    assert set(doc._annotation_graph.keys()) == {
        "entities1",
        "entities2",
        "crossrefs",
        "_artificial_root",
    }
    assert set(doc._annotation_graph["entities1"]) == {"texta"}
    assert set(doc._annotation_graph["entities2"]) == {"textb"}
    assert set(doc._annotation_graph["crossrefs"]) == {"texta", "textb"}
    assert set(doc._annotation_graph["_artificial_root"]) == {
        "entities1",
        "entities2",
        "crossrefs",
    }

    span1 = LabeledSpan(0, 2, label="a")
    assert span1.targets is None
    doc.entities1.append(span1)
    assert doc.entities1[0] == span1
    assert span1.target == doc.texta

    span2 = LabeledSpan(2, 4, label="b")
    assert span2.targets is None
    doc.entities2.append(span2)
    assert doc.entities2[0] == span2
    assert span2.target == doc.textb

    doublespan = DoubleTextSpan(0, 2, 1, 5)
    assert doublespan.targets is None
    doc.crossrefs.append(doublespan)
    assert doc.crossrefs[0] == doublespan
    assert doublespan.named_targets == {"text1": doc.texta, "text2": doc.textb}
    assert str(doublespan) == "te|ext2"


def test_annotation_list_with_named_targets_mismatch_error():
    @dataclasses.dataclass(eq=True, frozen=True)
    class TextSpan(Annotation):
        TARGET_NAMES = ("text",)
        start: int
        end: int

        def __str__(self) -> str:
            if self.targets is None:
                return ""
            text: str = self.named_targets["text"]  # type: ignore
            return str(text[self.start : self.end])

    @dataclasses.dataclass
    class TestDocument(Document):
        text: str
        entities1: AnnotationList[TextSpan] = annotation_field(named_targets={"textx": "text"})

    with pytest.raises(
        TypeError,
        match=re.escape("keys of targets ['textx'] do not match TextSpan.TARGET_NAMES ['text']"),
    ):
        doc = TestDocument(text="text1")


def test_annotation_list_with_missing_target_names():
    @dataclasses.dataclass
    class TestDocument(Document):
        texta: str
        textb: str
        # note that the entries in targets do not follow the order of DoubleTextSpan.TARGET_NAMES
        crossrefs: AnnotationList[DoubleTextSpan] = annotation_field(targets=["textb", "texta"])

    with pytest.raises(
        TypeError,
        match=re.escape(
            "A target name mapping is required for AnnotationLists containing Annotations with TARGET_NAMES, but "
            'AnnotationList "crossrefs" has no target_names. You should pass the named_targets dict containing the '
            "following keys (see Annotation \"DoubleTextSpan\") to annotation_field: ('text1', 'text2')"
        ),
    ):
        doc = TestDocument(texta="text1", textb="text2")


def test_annotation_list_number_of_targets_mismatch_error():
    @dataclasses.dataclass
    class TestDocument(Document):
        texta: str
        textb: str
        crossrefs: AnnotationList[DoubleTextSpan] = annotation_field(target="texta")

    with pytest.raises(
        TypeError,
        match=re.escape(
            "number of targets ['texta'] does not match number of entries in DoubleTextSpan.TARGET_NAMES: "
            "['text1', 'text2']"
        ),
    ):
        doc = TestDocument(texta="text1", textb="text2")


def test_annotation_list_artificial_root_error():
    @dataclasses.dataclass
    class TestDocument(Document):
        text: str
        _artificial_root: AnnotationList[LabeledSpan] = annotation_field(target="text")

    with pytest.raises(
        ValueError,
        match=re.escape(
            'Failed to add the "_artificial_root" node to the annotation graph because it already exists. Note '
            "that AnnotationList entries with that name are not allowed."
        ),
    ):
        doc = TestDocument(text="text1")
