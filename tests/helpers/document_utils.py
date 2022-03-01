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


def construct_document(
    text: str,
    tokens: Optional[List[str]] = None,
    entities: Optional[List[LabeledSpan]] = None,
    relations: Optional[List[BinaryRelation]] = None,
    sentences: Optional[List[LabeledSpan]] = None,
    entity_annotation_name: Optional[str] = "entities",
    relation_annotation_name: Optional[str] = "relations",
    sentence_annotation_name: Optional[str] = "sentences",
) -> Document:
    doc = Document(text=text)
    if tokens is not None:
        doc.metadata["tokens"] = tokens
    if sentences is not None:
        assert sentence_annotation_name is not None
        for i, sent in enumerate(sentences):
            _add_span(
                doc=doc,
                span=sent,
                annotation_name=sentence_annotation_name,
            )
    if entities is not None:
        assert entity_annotation_name is not None
        for i, ent in enumerate(entities):
            _add_span(
                doc=doc,
                span=ent,
                annotation_name=entity_annotation_name,
            )
    if relations is not None:
        assert relation_annotation_name is not None
        assert entities is not None
        for i, rel in enumerate(relations):
            _add_relation(
                doc=doc,
                rel=rel,
                annotation_name=relation_annotation_name,
            )

    return doc
