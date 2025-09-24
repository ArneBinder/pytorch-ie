import dataclasses
from typing import Dict

import pytest
from pie_core import Annotation, AnnotationLayer, Document, annotation_field
from pie_documents.annotations import BinaryRelation, LabeledMultiSpan, LabeledSpan, Span
from pie_documents.documents import TextBasedDocument, TokenBasedDocument
from transformers import AutoTokenizer, PreTrainedTokenizer

from pytorch_ie.utils.document import (
    SpanNotAlignedWithTokenException,
    get_aligned_token_span,
    tokenize_document,
)
from tests.conftest import TestDocument


@dataclasses.dataclass
class TokenizedTestDocument(TokenBasedDocument):
    sentences: AnnotationLayer[Span] = annotation_field(target="tokens")
    entities: AnnotationLayer[LabeledSpan] = annotation_field(target="tokens")
    relations: AnnotationLayer[BinaryRelation] = annotation_field(target="entities")


@dataclasses.dataclass
class TestDocumentWithMultiSpans(TextBasedDocument):
    entities: AnnotationLayer[LabeledMultiSpan] = annotation_field(target="text")
    relations: AnnotationLayer[BinaryRelation] = annotation_field(target="entities")


@dataclasses.dataclass
class TokenizedTestDocumentWithMultiSpans(TokenBasedDocument):
    entities: AnnotationLayer[LabeledMultiSpan] = annotation_field(target="tokens")
    relations: AnnotationLayer[BinaryRelation] = annotation_field(target="entities")


@pytest.fixture
def text_document():
    doc = TestDocument(text="First sentence. Entity M works at N. And it founded O.")
    doc.sentences.extend([Span(start=0, end=15), Span(start=16, end=36), Span(start=37, end=54)])
    doc.entities.extend(
        [
            LabeledSpan(start=16, end=24, label="per"),
            LabeledSpan(start=34, end=35, label="org"),
            LabeledSpan(start=41, end=43, label="per"),
            LabeledSpan(start=52, end=53, label="org"),
        ]
    )
    doc.relations.extend(
        [
            BinaryRelation(head=doc.entities[0], tail=doc.entities[1], label="per:employee_of"),
            BinaryRelation(head=doc.entities[2], tail=doc.entities[3], label="per:founder"),
        ]
    )
    _test_text_document(doc)
    return doc


def _test_text_document(doc):
    assert str(doc.sentences[0]) == "First sentence."
    assert str(doc.sentences[1]) == "Entity M works at N."
    assert str(doc.sentences[2]) == "And it founded O."

    assert str(doc.entities[0]) == "Entity M"
    assert str(doc.entities[1]) == "N"
    assert str(doc.entities[2]) == "it"
    assert str(doc.entities[3]) == "O"

    relation_tuples = [(str(rel.head), rel.label, str(rel.tail)) for rel in doc.relations]
    assert relation_tuples == [("Entity M", "per:employee_of", "N"), ("it", "per:founder", "O")]


@pytest.fixture
def text_document_with_multi_spans():
    doc = TestDocumentWithMultiSpans(text="First sentence. Entity M works at N. And it founded O.")
    doc.entities.extend(
        [
            LabeledMultiSpan(slices=((16, 22), (23, 24)), label="per"),
            LabeledMultiSpan(slices=((34, 35),), label="org"),
            LabeledMultiSpan(slices=((41, 43),), label="per"),
            LabeledMultiSpan(slices=((52, 53),), label="org"),
        ]
    )
    doc.relations.extend(
        [
            BinaryRelation(head=doc.entities[0], tail=doc.entities[1], label="per:employee_of"),
            BinaryRelation(head=doc.entities[2], tail=doc.entities[3], label="per:founder"),
        ]
    )
    _test_text_document_with_multi_spans(doc)
    return doc


def _test_text_document_with_multi_spans(doc):
    assert str(doc.entities[0]) == "('Entity', 'M')"
    assert str(doc.entities[1]) == "('N',)"
    assert str(doc.entities[2]) == "('it',)"
    assert str(doc.entities[3]) == "('O',)"

    relation_tuples = [(str(rel.head), rel.label, str(rel.tail)) for rel in doc.relations]
    assert relation_tuples == [
        ("('Entity', 'M')", "per:employee_of", "('N',)"),
        ("('it',)", "per:founder", "('O',)"),
    ]


@pytest.fixture
def token_document():
    doc = TokenizedTestDocument(
        tokens=(
            "[CLS]",
            "First",
            "sentence",
            ".",
            "Entity",
            "M",
            "works",
            "at",
            "N",
            ".",
            "And",
            "it",
            "founded",
            "O",
            ".",
            "[SEP]",
        ),
    )
    doc.sentences.extend(
        [
            Span(start=1, end=4),
            Span(start=4, end=10),
            Span(start=10, end=15),
        ]
    )
    doc.entities.extend(
        [
            LabeledSpan(start=4, end=6, label="per"),
            LabeledSpan(start=8, end=9, label="org"),
            LabeledSpan(start=11, end=12, label="per"),
            LabeledSpan(start=13, end=14, label="org"),
        ]
    )
    doc.relations.extend(
        [
            BinaryRelation(head=doc.entities[0], tail=doc.entities[1], label="per:employee_of"),
            BinaryRelation(head=doc.entities[2], tail=doc.entities[3], label="per:founder"),
        ]
    )
    _test_token_document(doc)
    return doc


def _test_token_document(doc):
    assert str(doc.sentences[0]) == "('First', 'sentence', '.')"
    assert str(doc.sentences[1]) == "('Entity', 'M', 'works', 'at', 'N', '.')"
    assert str(doc.sentences[2]) == "('And', 'it', 'founded', 'O', '.')"

    assert str(doc.entities[0]) == "('Entity', 'M')"
    assert str(doc.entities[1]) == "('N',)"
    assert str(doc.entities[2]) == "('it',)"
    assert str(doc.entities[3]) == "('O',)"

    relation_tuples = [(str(rel.head), rel.label, str(rel.tail)) for rel in doc.relations]
    assert relation_tuples == [
        ("('Entity', 'M')", "per:employee_of", "('N',)"),
        ("('it',)", "per:founder", "('O',)"),
    ]


@pytest.fixture
def token_document_with_multi_spans():
    doc = TokenizedTestDocumentWithMultiSpans(
        tokens=(
            "[CLS]",
            "First",
            "sentence",
            ".",
            "Entity",
            "M",
            "works",
            "at",
            "N",
            ".",
            "And",
            "it",
            "founded",
            "O",
            ".",
            "[SEP]",
        ),
    )
    doc.entities.extend(
        [
            LabeledMultiSpan(slices=((4, 5), (5, 6)), label="per"),
            LabeledMultiSpan(slices=((8, 9),), label="org"),
            LabeledMultiSpan(slices=((11, 12),), label="per"),
            LabeledMultiSpan(slices=((13, 14),), label="org"),
        ]
    )
    doc.relations.extend(
        [
            BinaryRelation(head=doc.entities[0], tail=doc.entities[1], label="per:employee_of"),
            BinaryRelation(head=doc.entities[2], tail=doc.entities[3], label="per:founder"),
        ]
    )
    _test_token_document_with_multi_spans(doc)
    return doc


def _test_token_document_with_multi_spans(doc):
    assert str(doc.entities[0]) == "(('Entity',), ('M',))"
    assert str(doc.entities[1]) == "(('N',),)"
    assert str(doc.entities[2]) == "(('it',),)"
    assert str(doc.entities[3]) == "(('O',),)"

    relation_tuples = [(str(rel.head), rel.label, str(rel.tail)) for rel in doc.relations]
    assert relation_tuples == [
        ("(('Entity',), ('M',))", "per:employee_of", "(('N',),)"),
        ("(('it',),)", "per:founder", "(('O',),)"),
    ]


@pytest.fixture(scope="module")
def tokenizer() -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained("bert-base-cased")


def _assert_added_annotations(
    document: Document,
    converted_document: Document,
    added_annotations: Dict[str, Dict[Annotation, Annotation]],
):
    for ann_field in document.annotation_fields():
        layer_name = ann_field.name
        text_annotations = document[layer_name]
        token_annotations = converted_document[layer_name]
        expected_mapping = dict(zip(text_annotations, token_annotations))
        assert len(expected_mapping) > 0
        assert added_annotations[layer_name] == expected_mapping


def test_tokenize_document(text_document, tokenizer):
    added_annotations = []
    tokenized_docs = tokenize_document(
        text_document,
        tokenizer=tokenizer,
        result_document_type=TokenizedTestDocument,
        added_annotations=added_annotations,
    )
    assert len(tokenized_docs) == 1
    tokenized_doc = tokenized_docs[0]

    # check (de-)serialization
    tokenized_doc.copy()

    assert (
        tokenized_doc.metadata["text"]
        == text_document.text
        == "First sentence. Entity M works at N. And it founded O."
    )
    assert tokenized_doc.tokens == (
        "[CLS]",
        "First",
        "sentence",
        ".",
        "En",
        "##ti",
        "##ty",
        "M",
        "works",
        "at",
        "N",
        ".",
        "And",
        "it",
        "founded",
        "O",
        ".",
        "[SEP]",
    )
    assert len(tokenized_doc.sentences) == len(text_document.sentences) == 3
    sentences = [str(sentence) for sentence in tokenized_doc.sentences]
    assert sentences == [
        "('First', 'sentence', '.')",
        "('En', '##ti', '##ty', 'M', 'works', 'at', 'N', '.')",
        "('And', 'it', 'founded', 'O', '.')",
    ]
    assert len(tokenized_doc.entities) == len(text_document.entities) == 4
    entities = [str(entity) for entity in tokenized_doc.entities]
    assert entities == ["('En', '##ti', '##ty', 'M')", "('N',)", "('it',)", "('O',)"]
    assert len(tokenized_doc.relations) == len(text_document.relations) == 2
    relation_tuples = [
        (str(rel.head), rel.label, str(rel.tail)) for rel in tokenized_doc.relations
    ]
    assert relation_tuples == [
        ("('En', '##ti', '##ty', 'M')", "per:employee_of", "('N',)"),
        ("('it',)", "per:founder", "('O',)"),
    ]

    assert len(added_annotations) == 1
    first_added_annotations = added_annotations[0]
    _assert_added_annotations(text_document, tokenized_doc, first_added_annotations)


def test_tokenize_document_max_length(text_document, tokenizer, caplog):
    added_annotations = []
    caplog.clear()
    with caplog.at_level("WARNING"):
        tokenized_docs = tokenize_document(
            text_document,
            tokenizer=tokenizer,
            result_document_type=TokenizedTestDocument,
            # max_length is set to 10, so the document is split into two parts
            strict_span_conversion=False,
            max_length=10,
            return_overflowing_tokens=True,
            added_annotations=added_annotations,
        )
    assert len(caplog.records) == 1
    assert (
        caplog.records[0].message
        == "could not convert all annotations from document with id=None to token based documents, missed annotations "
        "(disable this message with verbose=False):\n"
        "{\n"
        '  "relations": "{BinaryRelation(head=LabeledSpan(start=16, end=24, label=\'per\', score=1.0), '
        "tail=LabeledSpan(start=34, end=35, label='org', score=1.0), label='per:employee_of', score=1.0)}\",\n"
        '  "sentences": "{Span(start=16, end=36)}"\n'
        "}"
    )
    assert len(tokenized_docs) == 2
    assert len(added_annotations) == 2
    tokenized_doc = tokenized_docs[0]

    # check (de-)serialization
    tokenized_doc.copy()

    assert (
        tokenized_doc.metadata["text"]
        == text_document.text
        == "First sentence. Entity M works at N. And it founded O."
    )
    assert tokenized_doc.tokens == (
        "[CLS]",
        "First",
        "sentence",
        ".",
        "En",
        "##ti",
        "##ty",
        "M",
        "works",
        "[SEP]",
    )
    assert len(tokenized_doc.sentences) == 1
    sentences = [str(sentence) for sentence in tokenized_doc.sentences]
    assert sentences == ["('First', 'sentence', '.')"]
    assert len(tokenized_doc.entities) == 1
    entities = [str(entity) for entity in tokenized_doc.entities]
    assert entities == ["('En', '##ti', '##ty', 'M')"]
    assert len(tokenized_doc.relations) == 0
    # check annotation mapping
    current_added_annotations = added_annotations[0]
    # no relations are added in the first tokenized document
    assert set(current_added_annotations) == {"sentences", "entities"}
    # check sentences
    sentence_mapping = {
        k.resolve(): v.resolve() for k, v in current_added_annotations["sentences"].items()
    }
    assert sentence_mapping == {"First sentence.": ("First", "sentence", ".")}
    # check entities
    entity_mapping = {
        k.resolve(): v.resolve() for k, v in current_added_annotations["entities"].items()
    }
    assert entity_mapping == {("per", "Entity M"): ("per", ("En", "##ti", "##ty", "M"))}

    tokenized_doc = tokenized_docs[1]

    # check (de-)serialization
    tokenized_doc.copy()

    assert (
        tokenized_doc.metadata["text"]
        == text_document.text
        == "First sentence. Entity M works at N. And it founded O."
    )
    assert tokenized_doc.tokens == (
        "[CLS]",
        "at",
        "N",
        ".",
        "And",
        "it",
        "founded",
        "O",
        ".",
        "[SEP]",
    )
    assert len(tokenized_doc.sentences) == 1
    sentences = [str(sentence) for sentence in tokenized_doc.sentences]
    assert sentences == ["('And', 'it', 'founded', 'O', '.')"]
    assert len(tokenized_doc.entities) == 3
    entities = [str(entity) for entity in tokenized_doc.entities]
    assert entities == ["('N',)", "('it',)", "('O',)"]
    assert len(tokenized_doc.relations) == 1
    relation_tuples = [
        (str(rel.head), rel.label, str(rel.tail)) for rel in tokenized_doc.relations
    ]
    assert relation_tuples == [("('it',)", "per:founder", "('O',)")]
    # check annotation mapping
    current_added_annotations = added_annotations[1]
    assert set(current_added_annotations) == {"sentences", "entities", "relations"}
    # check sentences
    sentence_mapping = {
        k.resolve(): v.resolve() for k, v in current_added_annotations["sentences"].items()
    }
    assert sentence_mapping == {"And it founded O.": ("And", "it", "founded", "O", ".")}
    # check entities
    entity_mapping = {
        k.resolve(): v.resolve() for k, v in current_added_annotations["entities"].items()
    }
    assert entity_mapping == {
        ("org", "N"): ("org", ("N",)),
        ("per", "it"): ("per", ("it",)),
        ("org", "O"): ("org", ("O",)),
    }
    # check relations
    relation_mapping = {
        k.resolve(): v.resolve() for k, v in current_added_annotations["relations"].items()
    }
    assert relation_mapping == {
        ("per:founder", (("per", "it"), ("org", "O"))): (
            "per:founder",
            (("per", ("it",)), ("org", ("O",))),
        )
    }


def test_tokenize_document_max_length_strict(text_document, tokenizer):
    with pytest.raises(ValueError) as excinfo:
        tokenize_document(
            text_document,
            tokenizer=tokenizer,
            result_document_type=TokenizedTestDocument,
            # max_length is set to 10, so the document is split into two parts
            strict_span_conversion=True,
            max_length=10,
            return_overflowing_tokens=True,
        )
    assert (
        str(excinfo.value)
        == "could not convert all annotations from document with id=None to token based documents, "
        "but strict_span_conversion is True, so raise an error, missed annotations:\n"
        "{\n"
        '  "relations": "{BinaryRelation(head=LabeledSpan(start=16, end=24, label=\'per\', score=1.0), '
        "tail=LabeledSpan(start=34, end=35, label='org', score=1.0), label='per:employee_of', score=1.0)}\",\n"
        '  "sentences": "{Span(start=16, end=36)}"\n'
        "}"
    )


def test_tokenize_document_partition(text_document, tokenizer):
    added_annotations = []
    tokenized_docs = tokenize_document(
        text_document,
        tokenizer=tokenizer,
        result_document_type=TokenizedTestDocument,
        partition_layer="sentences",
        added_annotations=added_annotations,
    )
    assert len(tokenized_docs) == 3
    assert len(added_annotations) == 3
    tokenized_doc = tokenized_docs[0]

    # check (de-)serialization
    tokenized_doc.copy()

    assert (
        tokenized_doc.metadata["text"]
        == text_document.text
        == "First sentence. Entity M works at N. And it founded O."
    )
    assert tokenized_doc.tokens == ("[CLS]", "First", "sentence", ".", "[SEP]")
    assert len(tokenized_doc.sentences) == 1
    assert len(tokenized_doc.entities) == 0
    assert len(tokenized_doc.relations) == 0

    # check annotation mapping
    current_added_annotations = added_annotations[0]
    assert set(current_added_annotations) == {"sentences"}
    # check sentences
    sentence_mapping = {
        k.resolve(): v.resolve() for k, v in current_added_annotations["sentences"].items()
    }
    assert sentence_mapping == {"First sentence.": ("First", "sentence", ".")}

    tokenized_doc = tokenized_docs[1]

    # check (de-)serialization
    tokenized_doc.copy()

    assert (
        tokenized_doc.metadata["text"]
        == text_document.text
        == "First sentence. Entity M works at N. And it founded O."
    )
    assert tokenized_doc.tokens == (
        "[CLS]",
        "En",
        "##ti",
        "##ty",
        "M",
        "works",
        "at",
        "N",
        ".",
        "[SEP]",
    )
    assert len(tokenized_doc.sentences) == 1
    sentences = [str(sentence) for sentence in tokenized_doc.sentences]
    assert sentences == ["('En', '##ti', '##ty', 'M', 'works', 'at', 'N', '.')"]
    assert len(tokenized_doc.entities) == 2
    entities = [str(entity) for entity in tokenized_doc.entities]
    assert entities == ["('En', '##ti', '##ty', 'M')", "('N',)"]
    assert len(tokenized_doc.relations) == 1
    relation_tuples = [
        (str(rel.head), rel.label, str(rel.tail)) for rel in tokenized_doc.relations
    ]
    assert relation_tuples == [("('En', '##ti', '##ty', 'M')", "per:employee_of", "('N',)")]

    # check annotation mapping
    current_added_annotations = added_annotations[1]
    assert set(current_added_annotations) == {"sentences", "entities", "relations"}
    # check sentences
    sentence_mapping = {
        k.resolve(): v.resolve() for k, v in current_added_annotations["sentences"].items()
    }
    assert sentence_mapping == {
        "Entity M works at N.": ("En", "##ti", "##ty", "M", "works", "at", "N", ".")
    }
    # check entities
    entity_mapping = {
        k.resolve(): v.resolve() for k, v in current_added_annotations["entities"].items()
    }
    assert entity_mapping == {
        ("per", "Entity M"): ("per", ("En", "##ti", "##ty", "M")),
        ("org", "N"): ("org", ("N",)),
    }
    # check relations
    relation_mapping = {
        k.resolve(): v.resolve() for k, v in current_added_annotations["relations"].items()
    }
    assert relation_mapping == {
        ("per:employee_of", (("per", "Entity M"), ("org", "N"))): (
            "per:employee_of",
            (("per", ("En", "##ti", "##ty", "M")), ("org", ("N",))),
        )
    }

    tokenized_doc = tokenized_docs[2]

    # check (de-)serialization
    tokenized_doc.copy()

    assert (
        tokenized_doc.metadata["text"]
        == text_document.text
        == "First sentence. Entity M works at N. And it founded O."
    )
    assert tokenized_doc.tokens == ("[CLS]", "And", "it", "founded", "O", ".", "[SEP]")
    assert len(tokenized_doc.sentences) == 1
    sentences = [str(sentence) for sentence in tokenized_doc.sentences]
    assert sentences == ["('And', 'it', 'founded', 'O', '.')"]
    assert len(tokenized_doc.entities) == 2
    entities = [str(entity) for entity in tokenized_doc.entities]
    assert entities == ["('it',)", "('O',)"]
    assert len(tokenized_doc.relations) == 1
    relation_tuples = [
        (str(rel.head), rel.label, str(rel.tail)) for rel in tokenized_doc.relations
    ]
    assert relation_tuples == [("('it',)", "per:founder", "('O',)")]

    # check annotation mapping
    current_added_annotations = added_annotations[2]
    assert set(current_added_annotations) == {"sentences", "entities", "relations"}
    # check sentences
    sentence_mapping = {
        k.resolve(): v.resolve() for k, v in current_added_annotations["sentences"].items()
    }
    assert sentence_mapping == {"And it founded O.": ("And", "it", "founded", "O", ".")}
    # check entities
    entity_mapping = {
        k.resolve(): v.resolve() for k, v in current_added_annotations["entities"].items()
    }
    assert entity_mapping == {("per", "it"): ("per", ("it",)), ("org", "O"): ("org", ("O",))}
    # check relations
    relation_mapping = {
        k.resolve(): v.resolve() for k, v in current_added_annotations["relations"].items()
    }
    assert relation_mapping == {
        ("per:founder", (("per", "it"), ("org", "O"))): (
            "per:founder",
            (("per", ("it",)), ("org", ("O",))),
        )
    }


def test_get_aligned_token_span():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    text = "Hello, world!"
    encoding = tokenizer(text)
    tokens = tokenizer.convert_ids_to_tokens(encoding.input_ids)
    assert tokens == ["[CLS]", "Hello", ",", "world", "!", "[SEP]"]

    # already aligned
    char_span = Span(0, 5)
    assert text[char_span.start : char_span.end] == "Hello"
    token_span = get_aligned_token_span(encoding=encoding, char_span=char_span)
    assert tokens[token_span.start : token_span.end] == ["Hello"]

    # end not aligned
    char_span = Span(5, 7)
    assert text[char_span.start : char_span.end] == ", "
    token_span = get_aligned_token_span(encoding=encoding, char_span=char_span)
    assert tokens[token_span.start : token_span.end] == [","]

    # start not aligned
    char_span = Span(6, 12)
    assert text[char_span.start : char_span.end] == " world"
    token_span = get_aligned_token_span(encoding=encoding, char_span=char_span)
    assert tokens[token_span.start : token_span.end] == ["world"]

    # start not aligned, end inside token
    char_span = Span(6, 8)
    assert text[char_span.start : char_span.end] == " w"
    token_span = get_aligned_token_span(encoding=encoding, char_span=char_span)
    assert tokens[token_span.start : token_span.end] == ["world"]

    # empty char span
    char_span = Span(2, 2)
    assert text[char_span.start : char_span.end] == ""
    with pytest.raises(SpanNotAlignedWithTokenException) as e:
        get_aligned_token_span(encoding=encoding, char_span=char_span)
    assert e.value.span == char_span

    # empty token span
    char_span = Span(6, 7)
    assert text[char_span.start : char_span.end] == " "
    with pytest.raises(SpanNotAlignedWithTokenException) as e:
        get_aligned_token_span(encoding=encoding, char_span=char_span)
    assert e.value.span == char_span
