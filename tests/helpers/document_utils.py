from typing import List, Optional

from pytorch_ie import Document
from pytorch_ie.data import BinaryRelation, LabeledSpan


def _add_span(
    doc: Document,
    span: LabeledSpan,
    annotation_name: str,
    id: Optional[str] = None,
) -> LabeledSpan:
    assert doc.text[span.start : span.end] == span.metadata["text"]
    if id is not None:
        span.metadata["id"] = id
    doc.add_annotation(name=annotation_name, annotation=span)
    return span


def _add_relation(
    doc: Document,
    rel: BinaryRelation,
    annotation_name: str,
    id: Optional[str] = None,
) -> BinaryRelation:
    if id is not None:
        rel.metadata["id"] = id
    doc.add_annotation(name=annotation_name, annotation=rel)
    return rel


def _assert_span_text(doc: Document, span: LabeledSpan):
    assert doc.text[span.start : span.end] == span.metadata["text"]


def construct_document(
    text: str,
    doc_id: Optional[str] = None,
    tokens: Optional[List[str]] = None,
    entities: Optional[List[LabeledSpan]] = None,
    relations: Optional[List[BinaryRelation]] = None,
    sentences: Optional[List[LabeledSpan]] = None,
    entity_annotation_name: str = "entities",
    relation_annotation_name: str = "relations",
    sentence_annotation_name: str = "sentences",
    assert_span_text: bool = False,
) -> Document:
    doc = Document(text=text, doc_id=doc_id)
    if tokens is not None:
        doc.metadata["tokens"] = tokens
    if sentences is not None:
        doc.annotations.add_layer(name=sentence_annotation_name, annotations=sentences, annotation_type=LabeledSpan)
        if assert_span_text:
            for ann in doc.annotations[sentence_annotation_name]:
                _assert_span_text(doc, ann)
    if entities is not None:
        doc.annotations.add_layer(name=entity_annotation_name, annotations=entities, annotation_type=LabeledSpan)
        if assert_span_text:
            for ann in doc.annotations[entity_annotation_name]:
                _assert_span_text(doc, ann)
    if relations is not None:
        doc.annotations.add_layer(name=relation_annotation_name, annotations=relations, annotation_type=BinaryRelation)

    return doc
