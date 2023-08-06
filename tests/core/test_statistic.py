import dataclasses

import pytest

from pytorch_ie import DatasetDict
from pytorch_ie.annotations import LabeledSpan
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.core.statistic import flatten_dict
from pytorch_ie.documents import TextBasedDocument
from pytorch_ie.metrics.statistics import (
    DocumentFieldLengthCounter,
    DocumentSpanLengthCounter,
    DocumentSubFieldLengthCounter,
    DocumentTokenCounter,
    DummyCounter,
    LabelCounter,
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


def test_prepare_data(dataset):
    statistic = DummyCounter()
    values_nested = statistic(dataset)
    prepared_data = flatten_dict(values_nested)
    assert prepared_data == {
        ("train",): [1, 1, 1],
        ("test",): [1, 1, 1],
        ("validation",): [1, 1, 1],
    }
    statistic = LabelCounter(field="entities")
    values_nested = statistic(dataset)
    prepared_data = flatten_dict(values_nested)
    assert prepared_data == {
        ("train", "ORG"): [2],
        ("train", "MISC"): [3],
        ("train", "PER"): [2],
        ("train", "LOC"): [2],
        ("test", "LOC"): [2, 3],
        ("test", "PER"): [2, 2],
        ("validation", "ORG"): [2, 3],
        ("validation", "LOC"): [2],
        ("validation", "MISC"): [2],
        ("validation", "PER"): [2],
    }
    statistic = DocumentFieldLengthCounter(field="text")
    values_nested = statistic(dataset)
    prepared_data = flatten_dict(values_nested)
    assert prepared_data == {
        ("train",): [48, 15, 19],
        ("test",): [57, 11, 40],
        ("validation",): [65, 17, 187],
    }

    statistic = DocumentSpanLengthCounter(field="entities")
    values_nested = statistic(dataset)
    prepared_data = flatten_dict(values_nested)
    assert prepared_data == {
        ("train", "ORG"): [2],
        ("train", "MISC"): [6, 7],
        ("train", "PER"): [15],
        ("train", "LOC"): [8],
        ("test", "LOC"): [5, 6, 20],
        ("test", "PER"): [5, 11],
        ("validation", "ORG"): [14, 14, 8],
        ("validation", "LOC"): [6],
        ("validation", "MISC"): [11],
        ("validation", "PER"): [12],
    }

    # this is not super useful, we just collect teh lengths of the labels, but it is enough to test the code
    statistic = DocumentSubFieldLengthCounter(field="entities", subfield="label")
    values_nested = statistic(dataset)
    prepared_data = flatten_dict(values_nested)
    assert prepared_data == {
        ("train",): [3, 4, 4, 3, 3],
        ("test",): [3, 3, 3, 3, 3],
        ("validation",): [3, 3, 4, 3, 3, 3],
    }


@pytest.mark.slow
def test_prepare_data_tokenize(dataset):
    statistic = DocumentTokenCounter(
        field="text", tokenizer_name_or_path="bert-base-uncased", add_special_tokens=False
    )
    values_nested = statistic(dataset)
    prepared_data = flatten_dict(values_nested)
    assert prepared_data == {
        ("train",): [9, 2, 6],
        ("test",): [12, 4, 12],
        ("validation",): [11, 6, 38],
    }
