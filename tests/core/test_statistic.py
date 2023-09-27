import dataclasses

import pytest

from pytorch_ie import DatasetDict
from pytorch_ie.annotations import LabeledSpan
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.documents import TextBasedDocument, TokenBasedDocument
from pytorch_ie.metrics.statistics import (
    DummyCollector,
    FieldLengthCollector,
    LabelCountCollector,
    SpanLengthCollector,
    SubFieldLengthCollector,
    TokenCountCollector,
)
from tests import FIXTURES_ROOT


@pytest.fixture
def dataset():
    @dataclasses.dataclass
    class Conll2003Document(TextBasedDocument):
        entities: AnnotationList[LabeledSpan] = annotation_field(target="text")

    return DatasetDict.from_json(
        data_dir=FIXTURES_ROOT / "dataset_dict" / "conll2003_extract",
        document_type=Conll2003Document,
    )


def test_statistics(dataset):
    statistic = DummyCollector()
    values = statistic(dataset)
    assert values == {"train": {"sum": 3}, "test": {"sum": 3}, "validation": {"sum": 3}}

    statistic = LabelCountCollector(field="entities", labels=["LOC", "PER", "ORG", "MISC"])
    values = statistic(dataset)
    assert values == {
        "train": {
            "LOC": {
                "mean": 0.3333333333333333,
                "std": 0.4714045207910317,
                "min": 0,
                "max": 1,
                "len": 3,
                "sum": 1,
            },
            "PER": {
                "mean": 0.3333333333333333,
                "std": 0.4714045207910317,
                "min": 0,
                "max": 1,
                "len": 3,
                "sum": 1,
            },
            "ORG": {
                "mean": 0.3333333333333333,
                "std": 0.4714045207910317,
                "min": 0,
                "max": 1,
                "len": 3,
                "sum": 1,
            },
            "MISC": {
                "mean": 0.6666666666666666,
                "std": 0.9428090415820634,
                "min": 0,
                "max": 2,
                "len": 3,
                "sum": 2,
            },
        },
        "validation": {
            "LOC": {
                "mean": 0.3333333333333333,
                "std": 0.4714045207910317,
                "min": 0,
                "max": 1,
                "len": 3,
                "sum": 1,
            },
            "PER": {
                "mean": 0.3333333333333333,
                "std": 0.4714045207910317,
                "min": 0,
                "max": 1,
                "len": 3,
                "sum": 1,
            },
            "ORG": {"mean": 1.0, "std": 0.816496580927726, "min": 0, "max": 2, "len": 3, "sum": 3},
            "MISC": {
                "mean": 0.3333333333333333,
                "std": 0.4714045207910317,
                "min": 0,
                "max": 1,
                "len": 3,
                "sum": 1,
            },
        },
        "test": {
            "LOC": {"mean": 1.0, "std": 0.816496580927726, "min": 0, "max": 2, "len": 3, "sum": 3},
            "PER": {
                "mean": 0.6666666666666666,
                "std": 0.4714045207910317,
                "min": 0,
                "max": 1,
                "len": 3,
                "sum": 2,
            },
            "ORG": {"mean": 0.0, "std": 0.0, "min": 0, "max": 0, "len": 3, "sum": 0},
            "MISC": {"mean": 0.0, "std": 0.0, "min": 0, "max": 0, "len": 3, "sum": 0},
        },
    }

    statistic = LabelCountCollector(field="entities", labels="INFERRED")
    values = statistic(dataset)
    assert values == {
        "train": {
            "ORG": {"max": 1, "len": 1, "sum": 1},
            "MISC": {"max": 2, "len": 1, "sum": 2},
            "PER": {"max": 1, "len": 1, "sum": 1},
            "LOC": {"max": 1, "len": 1, "sum": 1},
        },
        "validation": {
            "ORG": {"max": 2, "len": 2, "sum": 3},
            "LOC": {"max": 1, "len": 1, "sum": 1},
            "MISC": {"max": 1, "len": 1, "sum": 1},
            "PER": {"max": 1, "len": 1, "sum": 1},
        },
        "test": {"LOC": {"max": 2, "len": 2, "sum": 3}, "PER": {"max": 1, "len": 2, "sum": 2}},
    }

    statistic = FieldLengthCollector(field="text")
    values = statistic(dataset)
    assert values == {
        "test": {"max": 57, "mean": 36.0, "min": 11, "std": 18.991226044325487},
        "train": {"max": 48, "mean": 27.333333333333332, "min": 15, "std": 14.70449666674185},
        "validation": {"max": 187, "mean": 89.66666666666667, "min": 17, "std": 71.5603863103665},
    }

    statistic = SpanLengthCollector(layer="entities")
    values = statistic(dataset)
    assert values == {
        "train": {"len": 5, "mean": 7.6, "std": 4.223742416388575, "min": 2, "max": 15},
        "validation": {
            "len": 6,
            "mean": 10.833333333333334,
            "std": 2.9674156357941426,
            "min": 6,
            "max": 14,
        },
        "test": {"len": 5, "mean": 9.4, "std": 5.748043145279966, "min": 5, "max": 20},
    }

    statistic = SpanLengthCollector(layer="entities", labels="INFERRED")
    values = statistic(dataset)
    assert values == {
        "train": {
            "ORG": {"max": 2, "len": 1},
            "MISC": {"max": 7, "len": 2},
            "PER": {"max": 15, "len": 1},
            "LOC": {"max": 8, "len": 1},
        },
        "test": {
            "LOC": {
                "max": 20,
                "len": 3,
            },
            "PER": {"max": 11, "len": 2},
        },
        "validation": {
            "ORG": {"max": 14, "len": 3},
            "LOC": {"max": 6, "len": 1},
            "MISC": {"max": 11, "len": 1},
            "PER": {"max": 12, "len": 1},
        },
    }

    # this is not super useful, we just collect teh lengths of the labels, but it is enough to test the code
    statistic = SubFieldLengthCollector(field="entities", subfield="label")
    values = statistic(dataset)
    assert values == {
        "test": {"max": 3, "mean": 3.0, "min": 3, "std": 0.0},
        "train": {"max": 4, "mean": 3.4, "min": 3, "std": 0.4898979485566356},
        "validation": {"max": 4, "mean": 3.1666666666666665, "min": 3, "std": 0.3726779962499649},
    }


@pytest.mark.slow
def test_statistics_with_tokenize(dataset):
    statistic = TokenCountCollector(
        text_field="text",
        tokenizer="bert-base-uncased",
        tokenizer_kwargs=dict(add_special_tokens=False),
    )
    values = statistic(dataset)
    assert values == {
        "test": {"max": 12, "mean": 9.333333333333334, "min": 4, "std": 3.7712361663282534},
        "train": {"max": 9, "mean": 5.666666666666667, "min": 2, "std": 2.8674417556808756},
        "validation": {"max": 38, "mean": 18.333333333333332, "min": 6, "std": 14.055445761538678},
    }

    @dataclasses.dataclass
    class TokenDocumentWithLabeledEntities(TokenBasedDocument):
        entities: AnnotationList[LabeledSpan] = annotation_field(target="tokens")

    statistic = SpanLengthCollector(
        layer="entities",
        tokenize=True,
        tokenizer="bert-base-uncased",
        tokenized_document_type=TokenDocumentWithLabeledEntities,
    )
    values = statistic(dataset)
    assert values == {
        "test": {"len": 5, "max": 4, "mean": 2.4, "min": 1, "std": 1.2000000000000002},
        "train": {"len": 5, "max": 2, "mean": 1.2, "min": 1, "std": 0.4},
        "validation": {
            "len": 6,
            "max": 2,
            "mean": 1.3333333333333333,
            "min": 1,
            "std": 0.4714045207910317,
        },
    }
