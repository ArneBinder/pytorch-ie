from typing import Optional

from pytorch_ie import Document
from pytorch_ie.data import BinaryRelation, LabeledSpan

TEXT_01 = "Jane lives in Berlin. this is no sentence about Karl\n"
TEXT_02 = "Seattle is a rainy city. Jenny Durkan is the city's mayor.\n"
TEXT_03 = "Karl enjoys sunny days in Berlin."

DOC1_ENTITY_JANE = LabeledSpan(start=0, end=4, label="person", metadata={"text": "Jane"})
DOC1_ENTITY_BERLIN = LabeledSpan(start=14, end=20, label="city", metadata={"text": "Berlin"})
DOC1_ENTITY_KARL = LabeledSpan(start=48, end=52, label="person", metadata={"text": "Karl"})
DOC1_SENTENCE1 = LabeledSpan(
    start=0, end=21, label="sentence", metadata={"text": "Jane lives in Berlin."}
)
DOC1_REL_LIVES_IN = BinaryRelation(
    head=DOC1_ENTITY_JANE, tail=DOC1_ENTITY_BERLIN, label="lives_in"
)

DOC2_ENTITY_SEATTLE = LabeledSpan(start=0, end=7, label="city", metadata={"text": "Seattle"})
DOC2_ENTITY_JENNY = LabeledSpan(
    start=25, end=37, label="person", metadata={"text": "Jenny Durkan"}
)
DOC2_SENTENCE1 = LabeledSpan(
    start=0, end=24, label="sentence", metadata={"text": "Seattle is a rainy city."}
)
DOC2_SENTENCE2 = LabeledSpan(
    start=25, end=58, label="sentence", metadata={"text": "Jenny Durkan is the city's mayor."}
)
DOC2_REL_MAYOR_OF = BinaryRelation(
    head=DOC2_ENTITY_JENNY, tail=DOC2_ENTITY_SEATTLE, label="mayor_of"
)

DOC3_ENTITY_KARL = LabeledSpan(start=0, end=4, label="person", metadata={"text": "Karl"})
DOC3_ENTITY_BERLIN = LabeledSpan(start=26, end=32, label="city", metadata={"text": "Berlin"})
DOC3_SENTENCE1 = LabeledSpan(
    start=0, end=33, label="sentence", metadata={"text": "Karl enjoys sunny days in Berlin."}
)

DOC1_TOKENS = [
    "[CLS]",
    "Jane",
    "lives",
    "in",
    "Berlin",
    ".",
    "this",
    "is",
    "no",
    "sentence",
    "about",
    "Karl",
    "[SEP]",
]
DOC2_TOKENS = [
    "[CLS]",
    "Seattle",
    "is",
    "a",
    "rainy",
    "city",
    ".",
    "Jenny",
    "Du",
    "##rka",
    "##n",
    "is",
    "the",
    "city",
    "'",
    "s",
    "mayor",
    ".",
    "[SEP]",
]
DOC3_TOKENS = ["[CLS]", "Karl", "enjoys", "sunny", "days", "in", "Berlin", ".", "[SEP]"]


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


def get_doc1(
    entity_annotation_name: str,
    relation_annotation_name: str,
    sentence_annotation_name: str,
    **kwargs,
) -> Document:
    doc = Document(text=TEXT_01)
    doc.metadata["tokens"] = DOC1_TOKENS

    for i, ent in enumerate([DOC1_ENTITY_JANE, DOC1_ENTITY_BERLIN, DOC1_ENTITY_KARL]):
        _add_span(
            doc=doc,
            span=ent,
            annotation_name=entity_annotation_name,
        )
    for i, sent in enumerate([DOC1_SENTENCE1]):
        _add_span(
            doc=doc,
            span=sent,
            annotation_name=sentence_annotation_name,
        )
    _add_relation(
        doc=doc,
        rel=DOC1_REL_LIVES_IN,
        annotation_name=relation_annotation_name,
    )
    return doc


def get_doc2(
    entity_annotation_name: str,
    relation_annotation_name: str,
    sentence_annotation_name: str,
) -> Document:
    doc = Document(text=TEXT_02)
    doc.metadata["tokens"] = DOC2_TOKENS

    for i, ent in enumerate([DOC2_ENTITY_SEATTLE, DOC2_ENTITY_JENNY]):
        _add_span(
            doc=doc,
            span=ent,
            annotation_name=entity_annotation_name,
        )
    for i, sent in enumerate([DOC2_SENTENCE1, DOC2_SENTENCE2]):
        _add_span(
            doc=doc,
            span=sent,
            annotation_name=sentence_annotation_name,
        )

    _add_relation(
        doc=doc,
        rel=DOC2_REL_MAYOR_OF,
        annotation_name=relation_annotation_name,
    )
    return doc


def get_doc3(
    entity_annotation_name: str,
    relation_annotation_name: str,
    sentence_annotation_name: str,
) -> Document:
    doc = Document(text=TEXT_03)
    doc.metadata["tokens"] = DOC3_TOKENS

    for i, ent in enumerate([DOC3_ENTITY_KARL, DOC3_ENTITY_BERLIN]):
        _add_span(
            doc=doc,
            span=ent,
            annotation_name=entity_annotation_name,
        )
    for i, sent in enumerate([DOC3_SENTENCE1]):
        _add_span(
            doc=doc,
            span=sent,
            annotation_name=sentence_annotation_name,
        )
    # TODO: this is kind of hacky
    doc._annotations[relation_annotation_name] = []
    return doc
