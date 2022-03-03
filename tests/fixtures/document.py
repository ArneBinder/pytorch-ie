from pytorch_ie import Document
from pytorch_ie.data import BinaryRelation, LabeledSpan
from pytorch_ie.data.document import construct_document

DOC1_TEXT = "Jane lives in Berlin. this is no sentence about Karl\n"
DOC2_TEXT = "Seattle is a rainy city. Jenny Durkan is the city's mayor.\n"
DOC3_TEXT = "Karl enjoys sunny days in Berlin."

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


def get_doc1(
    assert_span_text=True,
    **kwargs,
) -> Document:
    return construct_document(
        text=DOC1_TEXT,
        tokens=DOC1_TOKENS,
        spans=dict(
            sentences=[DOC1_SENTENCE1],
            entities=[DOC1_ENTITY_JANE, DOC1_ENTITY_BERLIN, DOC1_ENTITY_KARL],
        ),
        binary_relations=dict(
            relations=[DOC1_REL_LIVES_IN]
        ),
        assert_span_text=assert_span_text,
        **kwargs,
    )


def get_doc2(
    assert_span_text=True,
    **kwargs,
) -> Document:
    return construct_document(
        text=DOC2_TEXT,
        tokens=DOC2_TOKENS,
        spans=dict(
            sentences=[DOC2_SENTENCE1, DOC2_SENTENCE2],
            entities=[DOC2_ENTITY_SEATTLE, DOC2_ENTITY_JENNY],
        ),
        binary_relations=dict(
            relations=[DOC2_REL_MAYOR_OF]
        ),
        assert_span_text=assert_span_text,
        **kwargs,
    )


def get_doc3(
    assert_span_text=True,
    **kwargs
) -> Document:
    doc = construct_document(
        text=DOC3_TEXT,
        tokens=DOC3_TOKENS,
        spans=dict(
            sentences=[DOC3_SENTENCE1],
            entities=[DOC3_ENTITY_KARL, DOC3_ENTITY_BERLIN],
        ),
        binary_relations=dict(
            # We have to set relations with an empty list to create the
            # annotation layer at all. This is important for some implementations
            # of .encode_input() that assume that if relations are NOT available it is
            # in inference mode where it will create all possible entity pairs as
            # candidate relations.
            relations=[],
        ),
        assert_span_text=assert_span_text,
        **kwargs,
    )

    return doc
