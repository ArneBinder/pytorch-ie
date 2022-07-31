import re
from dataclasses import dataclass

import pytest

import datasets
from pytorch_ie.annotations import LabeledSpan, Span
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.documents import TextDocument


@pytest.fixture(scope="module")
def conll2003_test_split():
    # use test split since it is the smallest
    return datasets.load_dataset(
        path="pie/conll2003",
        split="test",
    )


@dataclass
class CoNLL2002Document(TextDocument):
    entities: AnnotationList[LabeledSpan] = annotation_field(target="text")


@dataclass
class DocumentWithParts(TextDocument):
    parts: AnnotationList[Span] = annotation_field(target="text")


@dataclass
class CoNLL2002WithPartsDocument(CoNLL2002Document, DocumentWithParts):
    pass


@dataclass
class DocumentWithEnts(TextDocument):
    ents: AnnotationList[LabeledSpan] = annotation_field(target="text")


@dataclass
class DocumentWithEntsWrongType(TextDocument):
    ents: AnnotationList[Span] = annotation_field(target="text")


@dataclass
class DocumentWithEntsAndParts(DocumentWithParts, DocumentWithEnts):
    pass


@dataclass
class DocumentWithPartsAndEntitiesSwapped(TextDocument):
    parts: AnnotationList[LabeledSpan] = annotation_field(target="text")
    entities: AnnotationList[Span] = annotation_field(target="text")


def _add_full_part(doc: DocumentWithParts) -> DocumentWithParts:
    doc.parts.append(Span(start=0, end=len(doc.text)))
    return doc


def test_as_document_type(conll2003_test_split):
    casted = conll2003_test_split.cast_document_type(CoNLL2002WithPartsDocument)
    with_parts = casted.map(lambda doc: _add_full_part(doc))
    assert "entities" in with_parts.column_names
    assert "parts" in with_parts.column_names
    doc0 = with_parts[0]
    assert set(doc0) == {"entities", "parts"}
    assert doc0.entities == conll2003_test_split[0].entities

    part0 = doc0.parts[0]
    assert isinstance(part0, Span)
    assert part0.start == 0
    assert part0.end == len(doc0.text)


def test_cast_document_type_remove_field(conll2003_test_split):
    casted = conll2003_test_split.cast_document_type(DocumentWithParts, allow_field_removal=True)
    with_partitions = casted.map(lambda doc: _add_full_part(doc))
    assert "entities" not in with_partitions.column_names
    assert "parts" in with_partitions.column_names
    doc0 = with_partitions[0]
    assert set(doc0) == {"parts"}

    part0 = doc0.parts[0]
    assert isinstance(part0, Span)
    assert part0.start == 0
    assert part0.end == len(doc0.text)


def test_cast_document_type_rename_field(conll2003_test_split):
    casted = conll2003_test_split.cast_document_type(
        DocumentWithEntsAndParts, field_mapping={"entities": "ents"}
    )
    with_parts = casted.map(lambda doc: _add_full_part(doc))
    assert "ents" in with_parts.column_names
    assert "parts" in with_parts.column_names
    doc0 = with_parts[0]
    assert set(doc0) == {"ents", "parts"}
    assert doc0.ents == conll2003_test_split[0].entities

    part0 = doc0.parts[0]
    assert isinstance(part0, Span)
    assert part0.start == 0
    assert part0.end == len(doc0.text)


def test_cast_document_type_swap_fields(conll2003_test_split):
    # just add "parts" to have another field to swap "entities" with
    casted = conll2003_test_split.cast_document_type(CoNLL2002WithPartsDocument)
    with_parts = casted.map(lambda doc: _add_full_part(doc))

    swapped = with_parts.cast_document_type(
        DocumentWithPartsAndEntitiesSwapped,
        field_mapping={"entities": "parts", "parts": "entities"},
    )
    assert "entities" in swapped.column_names
    assert "parts" in swapped.column_names
    doc0 = swapped[0]
    assert set(doc0) == {"entities", "parts"}
    assert doc0.parts == conll2003_test_split[0].entities

    part0 = doc0.entities[0]
    assert isinstance(part0, Span)
    assert part0.start == 0
    assert part0.end == len(doc0.text)


def test_cast_document_type_remove_field_not_allowed(conll2003_test_split):

    with pytest.raises(ValueError, match=re.escape('field "entities" of original document_type')):
        conll2003_test_split.cast_document_type(DocumentWithParts)


def test_cast_document_type_rename_wrong_type(conll2003_test_split):

    with pytest.raises(ValueError, match=re.escape("new field is not the same as old field:")):
        conll2003_test_split.cast_document_type(
            DocumentWithEntsWrongType, field_mapping={"entities": "ents"}
        )
