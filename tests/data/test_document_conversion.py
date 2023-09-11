import dataclasses

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizer

from pytorch_ie import (
    text_based_document_to_token_based,
    token_based_document_to_text_based,
    tokenize_document,
)
from pytorch_ie.annotations import BinaryRelation, LabeledSpan, Span
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.documents import TokenBasedDocument
from tests.conftest import TestDocument


@dataclasses.dataclass
class TokenizedTestDocument(TokenBasedDocument):
    sentences: AnnotationList[Span] = annotation_field(target="tokens")
    entities: AnnotationList[LabeledSpan] = annotation_field(target="tokens")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")


@pytest.fixture(scope="module")
def tokenizer() -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained("bert-base-cased")


def test_text_based_document_to_token_based(documents, tokenizer):
    assert len(documents) >= 3
    for i, doc in enumerate(documents[:3]):
        tokenized_text = tokenizer(doc.text, return_offsets_mapping=True)
        tokenized_doc = text_based_document_to_token_based(
            doc,
            tokens=tokenized_text.tokens(),
            result_document_type=TokenizedTestDocument,
            # to increase test coverage
            token_offset_mapping=None if i == 1 else tokenized_text.offset_mapping,
            # to increase test coverage
            char_to_token=None if i == 0 else tokenized_text.char_to_token,
        )
        assert tokenized_doc is not None

        # check (de-)serialization
        tokenized_doc.copy()

        if i == 0:
            assert doc.id == "train_doc1"
            assert tokenized_doc.metadata["text"] == doc.text == "A single sentence."
            assert tokenized_doc.metadata["token_offset_mapping"] == tokenized_text.offset_mapping
            assert tokenized_doc.metadata.get("char_to_token") is None
            assert tokenized_doc.tokens == ("[CLS]", "A", "single", "sentence", ".", "[SEP]")
            assert len(tokenized_doc.sentences) == len(doc.sentences) == 1
            assert str(doc.sentences[0]) == "A single sentence."
            assert str(tokenized_doc.sentences[0]) == "('A', 'single', 'sentence', '.')"
            assert len(tokenized_doc.entities) == len(doc.entities) == 0
            assert len(tokenized_doc.relations) == len(doc.relations) == 0
        elif i == 1:
            assert doc.id == "train_doc2"
            assert tokenized_doc.metadata["text"] == doc.text == "Entity A works at B."
            assert tokenized_doc.metadata.get("token_offset_mapping") is None
            assert tokenized_doc.metadata["char_to_token"] == tokenized_text.char_to_token
            assert tokenized_doc.tokens == (
                "[CLS]",
                "En",
                "##ti",
                "##ty",
                "A",
                "works",
                "at",
                "B",
                ".",
                "[SEP]",
            )
            assert len(tokenized_doc.sentences) == len(doc.sentences) == 1
            assert str(doc.sentences[0]) == "Entity A works at B."
            assert (
                str(tokenized_doc.sentences[0])
                == "('En', '##ti', '##ty', 'A', 'works', 'at', 'B', '.')"
            )
            assert len(tokenized_doc.entities) == len(doc.entities) == 2
            assert str(doc.entities[0]) == "Entity A"
            assert str(tokenized_doc.entities[0]) == "('En', '##ti', '##ty', 'A')"
            assert str(doc.entities[1]) == "B"
            assert str(tokenized_doc.entities[1]) == "('B',)"
            assert len(tokenized_doc.relations) == len(doc.relations) == 1
            assert doc.relations[0].head == doc.entities[0]
            assert tokenized_doc.relations[0].head == tokenized_doc.entities[0]
            assert doc.relations[0].tail == doc.entities[1]
            assert tokenized_doc.relations[0].tail == tokenized_doc.entities[1]
        elif i == 2:
            assert doc.id == "train_doc3"
            assert tokenized_doc.metadata["text"] == doc.text == "Entity C and D."
            assert tokenized_doc.metadata["token_offset_mapping"] == tokenized_text.offset_mapping
            assert tokenized_doc.metadata["char_to_token"] == tokenized_text.char_to_token
            assert tokenized_doc.tokens == (
                "[CLS]",
                "En",
                "##ti",
                "##ty",
                "C",
                "and",
                "D",
                ".",
                "[SEP]",
            )
            assert len(tokenized_doc.sentences) == len(doc.sentences) == 1
            assert str(doc.sentences[0]) == "Entity C and D."
            assert (
                str(tokenized_doc.sentences[0]) == "('En', '##ti', '##ty', 'C', 'and', 'D', '.')"
            )
            assert len(tokenized_doc.entities) == len(doc.entities) == 2
            assert str(doc.entities[0]) == "Entity C"
            assert str(tokenized_doc.entities[0]) == "('En', '##ti', '##ty', 'C')"
            assert str(doc.entities[1]) == "D"
            assert str(tokenized_doc.entities[1]) == "('D',)"
            assert len(tokenized_doc.relations) == len(doc.relations) == 0
        else:
            raise ValueError(f"Unexpected document: {doc.id}")


def test_text_based_document_to_token_based_missing_args(documents, tokenizer):
    with pytest.raises(ValueError) as excinfo:
        doc = documents[0]
        tokenized_text = tokenizer(doc.text)
        tokenized_doc = text_based_document_to_token_based(
            doc,
            tokens=tokenized_text.tokens(),
            result_document_type=TokenizedTestDocument,
        )
    assert (
        str(excinfo.value)
        == "either token_offset_mapping or char_to_token must be provided to convert a text based document "
        "to token based, but both are None"
    )


def test_text_based_document_to_token_based_unaligned_span_strict(documents, tokenizer):
    doc = documents[0].copy()
    # add a span that is not aligned with the tokenization
    doc.entities.append(LabeledSpan(start=0, end=2, label="unaligned"))
    assert str(doc.entities[-1]) == "A "
    tokenized_text = tokenizer(doc.text, return_offsets_mapping=True)
    with pytest.raises(ValueError) as excinfo:
        tokenized_doc = text_based_document_to_token_based(
            doc,
            tokens=tokenized_text.tokens(),
            result_document_type=TokenizedTestDocument,
            # to increase test coverage
            token_offset_mapping=tokenized_text.offset_mapping,
            # to increase test coverage
            char_to_token=tokenized_text.char_to_token,
        )
    assert (
        str(excinfo.value)
        == 'cannot find token span for character span: "A ", text="A single sentence.", '
        "token_offset_mapping=[(0, 0), (0, 1), (2, 8), (9, 17), (17, 18), (0, 0)]"
    )


def test_text_based_document_to_token_based_unaligned_span_not_strict(documents, tokenizer):
    doc = documents[0].copy()
    doc.entities.append(LabeledSpan(start=0, end=2, label="unaligned"))
    assert str(doc.entities[-1]) == "A "
    tokenized_text = tokenizer(doc.text, return_offsets_mapping=True)
    tokenized_doc = text_based_document_to_token_based(
        doc,
        tokens=tokenized_text.tokens(),
        result_document_type=TokenizedTestDocument,
        # to increase test coverage
        token_offset_mapping=tokenized_text.offset_mapping,
        # to increase test coverage
        char_to_token=tokenized_text.char_to_token,
        strict_span_conversion=False,
    )

    # check (de-)serialization
    tokenized_doc.copy()

    assert len(doc.entities) == 1
    # the unaligned span is not included in the tokenized document
    assert len(tokenized_doc.entities) == 0


@pytest.fixture
def token_documents(documents, tokenizer):
    result = []
    for doc in documents:
        tokenized_text = tokenizer(doc.text, return_offsets_mapping=True)
        tokenized_doc = text_based_document_to_token_based(
            doc,
            tokens=tokenized_text.tokens(),
            result_document_type=TokenizedTestDocument,
            char_to_token=tokenized_text.char_to_token,
            token_offset_mapping=tokenized_text.offset_mapping,
        )
        result.append(tokenized_doc)
    return result


def test_token_based_document_to_text_based(documents, token_documents):
    for doc, tokenized_doc in zip(documents, token_documents):
        reconstructed_doc = token_based_document_to_text_based(
            tokenized_doc,
            result_document_type=TestDocument,
        )
        assert reconstructed_doc is not None
        doc_dict = doc.asdict()
        reconstructed_doc_dict = reconstructed_doc.asdict()
        # remove all added metadata (original text, token_offset_mapping, char_to_token, tokens)
        reconstructed_doc_dict["metadata"] = {
            k: reconstructed_doc_dict["metadata"][k] for k in doc_dict["metadata"]
        }
        assert reconstructed_doc_dict == doc_dict


def test_token_based_document_to_text_based_with_join_tokens_with(documents):
    for doc in documents:
        # split the text by individual whitespace characters
        # so that we can reconstruct the original text via " ".join(tokens)
        tokens = []
        token_offset_mapping = []
        start = 0
        for token in doc.text.split(" "):
            tokens.append(token)
            end = start + len(token)
            token_offset_mapping.append((start, end))
            start = end + 1

        tokenized_doc = text_based_document_to_token_based(
            doc,
            tokens=tokens,
            result_document_type=TokenizedTestDocument,
            token_offset_mapping=token_offset_mapping,
        )
        reconstructed_doc = token_based_document_to_text_based(
            tokenized_doc,
            result_document_type=TestDocument,
            join_tokens_with=" ",
        )
        assert reconstructed_doc is not None
        assert reconstructed_doc.text == doc.text

        if doc.id in ["train_doc1", "train_doc7"]:
            doc_dict = doc.asdict()
            reconstructed_doc_dict = reconstructed_doc.asdict()
            # remove all added metadata (original text, token_offset_mapping, char_to_token, tokens)
            reconstructed_doc_dict["metadata"] = {
                k: reconstructed_doc_dict["metadata"][k] for k in doc_dict["metadata"]
            }
            assert reconstructed_doc_dict == doc_dict
        elif doc.id == "train_doc2":
            assert reconstructed_doc.sentences == doc.sentences
            assert len(reconstructed_doc.entities) == len(doc.entities) == 2
            assert str(reconstructed_doc.entities[0]) == str(doc.entities[0]) == "Entity A"
            assert str(doc.entities[1]) == "B"
            assert str(reconstructed_doc.entities[1]) == "B."
            assert len(reconstructed_doc.relations) == len(doc.relations) == 1
            assert (
                reconstructed_doc.relations[0].label == doc.relations[0].label == "per:employee_of"
            )
            assert doc.relations[0].head == doc.entities[0]
            assert reconstructed_doc.relations[0].head == reconstructed_doc.entities[0]
            assert doc.relations[0].tail == doc.entities[1]
            assert reconstructed_doc.relations[0].tail == reconstructed_doc.entities[1]
        elif doc.id == "train_doc3":
            assert reconstructed_doc.sentences == doc.sentences
            assert len(reconstructed_doc.entities) == len(doc.entities) == 2
            assert str(reconstructed_doc.entities[0]) == str(doc.entities[0]) == "Entity C"
            assert str(doc.entities[1]) == "D"
            assert str(reconstructed_doc.entities[1]) == "D."
            assert len(reconstructed_doc.relations) == len(doc.relations) == 0
        elif doc.id == "train_doc4":
            assert reconstructed_doc.sentences == doc.sentences
            assert len(reconstructed_doc.entities) == len(doc.entities) == 2
            assert str(reconstructed_doc.entities[0]) == str(doc.entities[0]) == "Entity E"
            assert str(doc.entities[1]) == "F"
            assert str(reconstructed_doc.entities[1]) == "F."
            assert len(reconstructed_doc.relations) == len(doc.relations) == 0
        elif doc.id == "train_doc5":
            assert reconstructed_doc.sentences == doc.sentences
            assert len(reconstructed_doc.entities) == len(doc.entities) == 3
            assert str(reconstructed_doc.entities[0]) == str(doc.entities[0]) == "Entity G"
            assert str(doc.entities[1]) == "H"
            assert str(reconstructed_doc.entities[1]) == "H."
            assert str(doc.entities[2]) == "I"
            assert str(reconstructed_doc.entities[2]) == "I."
            assert len(reconstructed_doc.relations) == len(doc.relations) == 3
            assert (
                reconstructed_doc.relations[0].label == doc.relations[0].label == "per:employee_of"
            )
            assert doc.relations[0].head == doc.entities[0]
            assert reconstructed_doc.relations[0].head == reconstructed_doc.entities[0]
            assert doc.relations[0].tail == doc.entities[1]
            assert reconstructed_doc.relations[0].tail == reconstructed_doc.entities[1]
            assert reconstructed_doc.relations[1].label == doc.relations[1].label == "per:founder"
            assert doc.relations[1].head == doc.entities[0]
            assert reconstructed_doc.relations[1].head == reconstructed_doc.entities[0]
            assert doc.relations[1].tail == doc.entities[2]
            assert reconstructed_doc.relations[1].tail == reconstructed_doc.entities[2]
            assert (
                reconstructed_doc.relations[2].label == doc.relations[2].label == "org:founded_by"
            )
            assert doc.relations[2].head == doc.entities[2]
            assert reconstructed_doc.relations[2].head == reconstructed_doc.entities[2]
            assert doc.relations[2].tail == doc.entities[1]
            assert reconstructed_doc.relations[2].tail == reconstructed_doc.entities[1]
        elif doc.id == "train_doc6":
            assert reconstructed_doc.sentences == doc.sentences
            assert len(reconstructed_doc.entities) == len(doc.entities) == 3
            assert str(doc.entities[0]) == "Entity J"
            assert str(reconstructed_doc.entities[0]) == "Entity J,"
            assert str(doc.entities[1]) == "K"
            assert str(reconstructed_doc.entities[1]) == "K,"
            assert str(doc.entities[2]) == "L"
            assert str(reconstructed_doc.entities[2]) == "L."
            assert len(reconstructed_doc.relations) == len(doc.relations) == 0
        elif doc.id == "train_doc8":
            assert len(reconstructed_doc.sentences) == len(doc.sentences) == 3
            assert (
                str(reconstructed_doc.sentences[0]) == str(doc.sentences[0]) == "First sentence."
            )
            assert (
                str(reconstructed_doc.sentences[1])
                == str(doc.sentences[1])
                == "Entity M works at N."
            )
            assert str(doc.sentences[2]) == "And it founded O"
            assert str(reconstructed_doc.sentences[2]) == "And it founded O."
            assert len(reconstructed_doc.entities) == len(doc.entities) == 4
            assert str(reconstructed_doc.entities[0]) == str(doc.entities[0]) == "Entity M"
            assert str(doc.entities[1]) == "N"
            assert str(reconstructed_doc.entities[1]) == "N."
            assert str(reconstructed_doc.entities[2]) == str(doc.entities[2]) == "it"
            assert str(doc.entities[3]) == "O"
            assert str(reconstructed_doc.entities[3]) == "O."
            assert len(reconstructed_doc.relations) == len(doc.relations) == 3
            assert (
                reconstructed_doc.relations[0].label == doc.relations[0].label == "per:employee_of"
            )
            assert doc.relations[0].head == doc.entities[0]
            assert reconstructed_doc.relations[0].head == reconstructed_doc.entities[0]
            assert doc.relations[0].tail == doc.entities[1]
            assert reconstructed_doc.relations[0].tail == reconstructed_doc.entities[1]
            assert reconstructed_doc.relations[1].label == doc.relations[1].label == "per:founder"
            assert doc.relations[1].head == doc.entities[2]
            assert reconstructed_doc.relations[1].head == reconstructed_doc.entities[2]
            assert doc.relations[1].tail == doc.entities[3]
            assert reconstructed_doc.relations[1].tail == reconstructed_doc.entities[3]
            assert (
                reconstructed_doc.relations[2].label == doc.relations[2].label == "org:founded_by"
            )
            assert doc.relations[2].head == doc.entities[3]
            assert reconstructed_doc.relations[2].head == reconstructed_doc.entities[3]
            assert doc.relations[2].tail == doc.entities[2]
            assert reconstructed_doc.relations[2].tail == reconstructed_doc.entities[2]
        else:
            raise ValueError(f"Unexpected document: {doc.id}")


def test_tokenize_document(documents, tokenizer):
    doc = documents[1]
    tokenized_docs = tokenize_document(
        doc,
        tokenizer=tokenizer,
        result_document_type=TokenizedTestDocument,
    )
    assert len(tokenized_docs) == 1
    tokenized_doc = tokenized_docs[0]

    # check (de-)serialization
    tokenized_doc.copy()

    assert doc.id == "train_doc2"
    assert tokenized_doc.metadata["text"] == doc.text == "Entity A works at B."
    assert tokenized_doc.tokens == (
        "[CLS]",
        "En",
        "##ti",
        "##ty",
        "A",
        "works",
        "at",
        "B",
        ".",
        "[SEP]",
    )
    assert len(tokenized_doc.sentences) == len(doc.sentences) == 1
    assert str(doc.sentences[0]) == "Entity A works at B."
    assert (
        str(tokenized_doc.sentences[0]) == "('En', '##ti', '##ty', 'A', 'works', 'at', 'B', '.')"
    )
    assert len(tokenized_doc.entities) == len(doc.entities) == 2
    assert str(doc.entities[0]) == "Entity A"
    assert str(tokenized_doc.entities[0]) == "('En', '##ti', '##ty', 'A')"
    assert str(doc.entities[1]) == "B"
    assert str(tokenized_doc.entities[1]) == "('B',)"
    assert len(tokenized_doc.relations) == len(doc.relations) == 1
    assert tokenized_doc.relations[0].label == doc.relations[0].label == "per:employee_of"
    assert doc.relations[0].head == doc.entities[0]
    assert tokenized_doc.relations[0].head == tokenized_doc.entities[0]
    assert doc.relations[0].tail == doc.entities[1]
    assert tokenized_doc.relations[0].tail == tokenized_doc.entities[1]


def test_tokenize_document_max_length(documents, tokenizer):
    doc = documents[1]
    assert doc.id == "train_doc2"
    assert doc.text == "Entity A works at B."
    assert len(doc.sentences) == 1
    assert str(doc.sentences[0]) == "Entity A works at B."
    assert len(doc.entities) == 2
    assert str(doc.entities[0]) == "Entity A"
    assert str(doc.entities[1]) == "B"
    assert len(doc.relations) == 1
    assert doc.relations[0].label == "per:employee_of"
    assert doc.relations[0].head == doc.entities[0]
    assert doc.relations[0].tail == doc.entities[1]

    tokenized_docs = tokenize_document(
        doc,
        tokenizer=tokenizer,
        result_document_type=TokenizedTestDocument,
        strict_span_conversion=False,
        # This will cut out the second entity. Also, the sentence annotation will be removed,
        # because the sentence is not complete anymore.
        max_length=8,
        return_overflowing_tokens=True,
    )
    assert len(tokenized_docs) == 2
    tokenized_doc = tokenized_docs[0]

    # check (de-)serialization
    tokenized_doc.copy()

    assert tokenized_doc.id == doc.id == "train_doc2"
    assert tokenized_doc.metadata["text"] == doc.text == "Entity A works at B."
    assert tokenized_doc.tokens == ("[CLS]", "En", "##ti", "##ty", "A", "works", "at", "[SEP]")
    assert len(tokenized_doc.sentences) == 0
    assert len(tokenized_doc.entities) == 1
    assert str(tokenized_doc.entities[0]) == "('En', '##ti', '##ty', 'A')"
    assert len(tokenized_doc.relations) == 0

    tokenized_doc = tokenized_docs[1]

    # check (de-)serialization
    tokenized_doc.copy()

    assert tokenized_doc.id == doc.id == "train_doc2"
    assert tokenized_doc.metadata["text"] == doc.text == "Entity A works at B."
    assert tokenized_doc.tokens == ("[CLS]", "B", ".", "[SEP]")
    assert len(tokenized_doc.sentences) == 0
    assert len(tokenized_doc.entities) == 1
    assert str(tokenized_doc.entities[0]) == "('B',)"
    assert len(tokenized_doc.relations) == 0


def test_tokenize_document_partition(documents, tokenizer):
    doc = documents[7]
    assert doc.id == "train_doc8"
    assert doc.text == "First sentence. Entity M works at N. And it founded O."
    assert len(doc.sentences) == 3
    assert str(doc.sentences[0]) == "First sentence."
    assert str(doc.sentences[1]) == "Entity M works at N."
    assert str(doc.sentences[2]) == "And it founded O"
    assert len(doc.entities) == 4
    assert str(doc.entities[0]) == "Entity M"
    assert str(doc.entities[1]) == "N"
    assert str(doc.entities[2]) == "it"
    assert str(doc.entities[3]) == "O"
    assert len(doc.relations) == 3
    assert doc.relations[0].head == doc.entities[0]
    assert doc.relations[0].tail == doc.entities[1]
    assert doc.relations[1].head == doc.entities[2]
    assert doc.relations[1].tail == doc.entities[3]
    assert doc.relations[2].head == doc.entities[3]
    assert doc.relations[2].tail == doc.entities[2]

    tokenized_docs = tokenize_document(
        doc,
        tokenizer=tokenizer,
        result_document_type=TokenizedTestDocument,
        strict_span_conversion=False,
        partition_layer="sentences",
    )
    assert len(tokenized_docs) == 3
    tokenized_doc = tokenized_docs[0]

    # check (de-)serialization
    tokenized_doc.copy()

    assert tokenized_doc.id == doc.id == "train_doc8"
    assert (
        tokenized_doc.metadata["text"]
        == doc.text
        == "First sentence. Entity M works at N. And it founded O."
    )
    assert tokenized_doc.tokens == ("[CLS]", "First", "sentence", ".", "[SEP]")
    assert len(tokenized_doc.sentences) == 1
    assert len(tokenized_doc.entities) == 0
    assert len(tokenized_doc.relations) == 0

    tokenized_doc = tokenized_docs[1]

    # check (de-)serialization
    tokenized_doc.copy()

    assert tokenized_doc.id == doc.id == "train_doc8"
    assert (
        tokenized_doc.metadata["text"]
        == doc.text
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
    assert len(tokenized_doc.entities) == 2
    assert str(tokenized_doc.entities[0]) == "('En', '##ti', '##ty', 'M')"
    assert str(tokenized_doc.entities[1]) == "('N',)"
    assert len(tokenized_doc.relations) == 1
    assert tokenized_doc.relations[0].label == "per:employee_of"
    assert tokenized_doc.relations[0].head == tokenized_doc.entities[0]
    assert tokenized_doc.relations[0].tail == tokenized_doc.entities[1]

    tokenized_doc = tokenized_docs[2]

    # check (de-)serialization
    tokenized_doc.copy()

    assert tokenized_doc.id == doc.id == "train_doc8"
    assert (
        tokenized_doc.metadata["text"]
        == doc.text
        == "First sentence. Entity M works at N. And it founded O."
    )
    assert tokenized_doc.tokens == ("[CLS]", "And", "it", "founded", "O", "[SEP]")
    assert len(tokenized_doc.sentences) == 1
    assert len(tokenized_doc.entities) == 2
    assert str(tokenized_doc.entities[0]) == "('it',)"
    assert str(tokenized_doc.entities[1]) == "('O',)"
    assert len(tokenized_doc.relations) == 2
    assert tokenized_doc.relations[0].label == "per:founder"
    assert tokenized_doc.relations[0].head == tokenized_doc.entities[0]
    assert tokenized_doc.relations[0].tail == tokenized_doc.entities[1]
    assert tokenized_doc.relations[1].label == "org:founded_by"
    assert tokenized_doc.relations[1].head == tokenized_doc.entities[1]
    assert tokenized_doc.relations[1].tail == tokenized_doc.entities[0]
