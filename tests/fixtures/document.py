from typing import List, Optional

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
    for i, sent in enumerate(sentences or []):
        _add_span(
            doc=doc,
            span=sent,
            annotation_name=sentence_annotation_name,
        )
    for i, ent in enumerate(entities or []):
        _add_span(
            doc=doc,
            span=ent,
            annotation_name=entity_annotation_name,
        )
    for i, rel in enumerate(relations or []):
        _add_relation(
            doc=doc,
            rel=rel,
            annotation_name=relation_annotation_name,
        )

    return doc


def get_doc1(
    **kwargs,
) -> Document:
    return construct_document(
        text=TEXT_01,
        tokens=DOC1_TOKENS,
        sentences=[DOC1_SENTENCE1],
        entities=[DOC1_ENTITY_JANE, DOC1_ENTITY_BERLIN, DOC1_ENTITY_KARL],
        relations=[DOC1_REL_LIVES_IN],
        **kwargs,
    )


def get_doc2(
    **kwargs,
) -> Document:
    return construct_document(
        text=TEXT_02,
        tokens=DOC2_TOKENS,
        sentences=[DOC2_SENTENCE1, DOC2_SENTENCE2],
        entities=[DOC2_ENTITY_SEATTLE, DOC2_ENTITY_JENNY],
        relations=[DOC2_REL_MAYOR_OF],
        **kwargs,
    )


def get_doc3(relation_annotation_name: Optional[str] = "relations", **kwargs) -> Document:
    doc = construct_document(
        text=TEXT_03,
        tokens=DOC3_TOKENS,
        sentences=[DOC3_SENTENCE1],
        entities=[DOC3_ENTITY_KARL, DOC3_ENTITY_BERLIN],
        relation_annotation_name=relation_annotation_name,
        **kwargs,
    )

    # TODO: this is kind of hacky
    doc._annotations[relation_annotation_name] = []
    return doc
