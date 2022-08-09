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


def test_cast_document_type(conll2003_test_split):
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
    casted = conll2003_test_split.cast_document_type(DocumentWithParts, remove_columns=True)
    with_partitions = casted.map(lambda doc: _add_full_part(doc))
    assert "entities" not in with_partitions.column_names
    assert "parts" in with_partitions.column_names
    doc0 = with_partitions[0]
    assert set(doc0) == {"parts"}

    part0 = doc0.parts[0]
    assert isinstance(part0, Span)
    assert part0.start == 0
    assert part0.end == len(doc0.text)

    casted_back = with_partitions.cast_document_type(CoNLL2002Document)
    assert "entities" in casted_back.column_names
    # original entities are not available anymore after casting back
    assert len(conll2003_test_split[0].entities) > 0
    assert len(casted_back[0].entities) == 0


def test_cast_document_type_recover_field(conll2003_test_split):
    doc_orig = conll2003_test_split[0]
    casted = conll2003_test_split.cast_document_type(DocumentWithParts)
    # "entities" stay in the arrow table because remove_columns=False per default
    assert "entities" in casted.column_names
    assert "parts" in casted.column_names

    doc_casted = casted[0]
    assert set(doc_casted) == {"parts"}

    casted_back = casted.cast_document_type(CoNLL2002Document)
    assert "entities" in casted_back.column_names
    # original entities are recovered after casting back
    doc_back = casted_back[0]
    assert len(doc_back.entities) > 0
    assert doc_back.entities == doc_orig.entities


def test_cast_document_type_recover_field_with_mapping(conll2003_test_split):
    doc_orig = conll2003_test_split[0]
    casted = conll2003_test_split.cast_document_type(DocumentWithParts)
    # "entities" stay in the arrow table because remove_columns=False per default
    assert "entities" in casted.column_names
    assert "parts" in casted.column_names

    doc_casted = casted[0]
    assert set(doc_casted) == {"parts"}

    casted_back = casted.cast_document_type(
        DocumentWithEntsAndParts, field_mapping={"entities": "ents"}
    )
    assert "ents" in casted_back.column_names
    # original entities are recovered after casting back
    doc_back = casted_back[0]
    assert len(doc_back.ents) > 0
    assert doc_back.ents == doc_orig.entities


def test_cast_document_type_recover_field_wrong(conll2003_test_split):
    doc_orig = conll2003_test_split[0]
    casted = conll2003_test_split.cast_document_type(DocumentWithEntsAndParts)
    # "entities" stay in the arrow table because remove_columns=False per default
    assert "entities" in casted.column_names
    assert "parts" in casted.column_names
    assert "ents" in casted.column_names

    doc_casted = casted[0]
    assert set(doc_casted) == {"parts", "ents"}

    with pytest.raises(
        ValueError,
        match=re.escape(
            "rename targets are already in column names: {'entities'}. Did you miss to set remove_columns=True in a previous call of cast_document_type?"
        ),
    ):
        casted.cast_document_type(CoNLL2002Document, field_mapping={"ents": "entities"})


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
    doc_with_parts = with_parts[0]

    swapped = with_parts.cast_document_type(
        DocumentWithPartsAndEntitiesSwapped,
        field_mapping={"entities": "parts", "parts": "entities"},
    )
    assert "entities" in swapped.column_names
    assert "parts" in swapped.column_names
    doc_swapped = swapped[0]
    assert set(doc_swapped) == {"entities", "parts"}
    assert doc_swapped.parts == doc_with_parts.entities
    assert doc_swapped.entities == doc_with_parts.parts


def test_cast_document_type_rename_source_not_available(conll2003_test_split):

    with pytest.raises(
        ValueError,
        match=re.escape(
            "some fields to rename are not in the original document_type or hidden fields: {'not_in_original_document'}"
        ),
    ):
        conll2003_test_split.cast_document_type(
            DocumentWithEntsWrongType, field_mapping={"not_in_original_document": "ents"}
        )


def test_cast_document_type_rename_target_not_available(conll2003_test_split):

    with pytest.raises(
        ValueError,
        match=re.escape(
            "some renamed fields are not in the new document_type: {'not_in_new_document'}"
        ),
    ):
        conll2003_test_split.cast_document_type(
            DocumentWithEntsWrongType, field_mapping={"entities": "not_in_new_document"}
        )


def test_cast_document_type_rename_wrong_type(conll2003_test_split):

    with pytest.raises(ValueError, match=re.escape("new field is not the same as old field:")):
        conll2003_test_split.cast_document_type(
            DocumentWithEntsWrongType, field_mapping={"entities": "ents"}
        )
