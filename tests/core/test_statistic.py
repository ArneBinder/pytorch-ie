import dataclasses

import pytest

from pytorch_ie import DatasetDict
from pytorch_ie.annotations import LabeledSpan
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.documents import TextBasedDocument
from pytorch_ie.metrics.statistics import (
    DummyCollector,
    FieldLengthCollector,
    LabelCountCollector,
    LabeledSpanLengthCollector,
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
        "test": {
            "LOC": {"len": 3, "max": 2, "mean": 1.0, "min": 0, "std": 0.816496580927726},
            "MISC": {"len": 3, "max": 0, "mean": 0.0, "min": 0, "std": 0.0},
            "ORG": {"len": 3, "max": 0, "mean": 0.0, "min": 0, "std": 0.0},
            "PER": {
                "len": 3,
                "max": 1,
                "mean": 0.6666666666666666,
                "min": 0,
                "std": 0.4714045207910317,
            },
        },
        "train": {
            "LOC": {
                "len": 3,
                "max": 1,
                "mean": 0.3333333333333333,
                "min": 0,
                "std": 0.4714045207910317,
            },
            "MISC": {
                "len": 3,
                "max": 2,
                "mean": 0.6666666666666666,
                "min": 0,
                "std": 0.9428090415820634,
            },
            "ORG": {
                "len": 3,
                "max": 1,
                "mean": 0.3333333333333333,
                "min": 0,
                "std": 0.4714045207910317,
            },
            "PER": {
                "len": 3,
                "max": 1,
                "mean": 0.3333333333333333,
                "min": 0,
                "std": 0.4714045207910317,
            },
        },
        "validation": {
            "LOC": {
                "len": 3,
                "max": 1,
                "mean": 0.3333333333333333,
                "min": 0,
                "std": 0.4714045207910317,
            },
            "MISC": {
                "len": 3,
                "max": 1,
                "mean": 0.3333333333333333,
                "min": 0,
                "std": 0.4714045207910317,
            },
            "ORG": {"len": 3, "max": 2, "mean": 1.0, "min": 0, "std": 0.816496580927726},
            "PER": {
                "len": 3,
                "max": 1,
                "mean": 0.3333333333333333,
                "min": 0,
                "std": 0.4714045207910317,
            },
        },
    }

    statistic = FieldLengthCollector(field="text")
    values = statistic(dataset)
    assert values == {
        "test": {"max": 57, "mean": 36.0, "min": 11, "std": 18.991226044325487},
        "train": {"max": 48, "mean": 27.333333333333332, "min": 15, "std": 14.70449666674185},
        "validation": {"max": 187, "mean": 89.66666666666667, "min": 17, "std": 71.5603863103665},
    }

    statistic = LabeledSpanLengthCollector(field="entities")
    values = statistic(dataset)
    assert values == {
        "train": {
            "ORG": {"mean": 2.0, "std": 0.0, "min": 2, "max": 2, "len": 1},
            "MISC": {"mean": 6.5, "std": 0.5, "min": 6, "max": 7, "len": 2},
            "PER": {"mean": 15.0, "std": 0.0, "min": 15, "max": 15, "len": 1},
            "LOC": {"mean": 8.0, "std": 0.0, "min": 8, "max": 8, "len": 1},
        },
        "test": {
            "LOC": {
                "mean": 10.333333333333334,
                "std": 6.847546194724712,
                "min": 5,
                "max": 20,
                "len": 3,
            },
            "PER": {"mean": 8.0, "std": 3.0, "min": 5, "max": 11, "len": 2},
        },
        "validation": {
            "ORG": {"mean": 12.0, "std": 2.8284271247461903, "min": 8, "max": 14, "len": 3},
            "LOC": {"mean": 6.0, "std": 0.0, "min": 6, "max": 6, "len": 1},
            "MISC": {"mean": 11.0, "std": 0.0, "min": 11, "max": 11, "len": 1},
            "PER": {"mean": 12.0, "std": 0.0, "min": 12, "max": 12, "len": 1},
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
