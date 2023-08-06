import dataclasses
from collections import defaultdict
from functools import partial
from typing import Callable, Dict, List, Tuple, Union

import pytest
from typing_extensions import TypeAlias

from pytorch_ie import DatasetDict
from pytorch_ie.annotations import LabeledSpan
from pytorch_ie.core import AnnotationList, Document, annotation_field
from pytorch_ie.core.statistic import flatten_dict
from pytorch_ie.documents import TextBasedDocument
from pytorch_ie.metrics.statistics import (
    DocumentFieldLengthCounter,
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


def label_counter(doc: Document, field: str) -> Dict[str, int]:
    field_obj = getattr(doc, field)
    counts: Dict[str, int] = defaultdict(lambda: 1)
    for elem in field_obj:
        counts[elem.label] += 1
    return dict(counts)


def document_field_length_collector(doc: Document, field: str) -> int:
    field_obj = getattr(doc, field)
    return len(field_obj)


# The metric should return a single int or float or a list of such values ...
BaseType: TypeAlias = Union[int, float]
ResultTerminal: TypeAlias = Union[BaseType, List[BaseType]]
# ... or such entries nested arbitrarily deep inside dictionaries.
ResultDict: TypeAlias = Dict[str, Union[ResultTerminal, "ResultDict"]]


def prepare_data(
    dataset: DatasetDict,
    metric: Callable[[Document], Union[ResultTerminal, ResultDict]],
) -> Dict[Tuple[str, ...], List[BaseType]]:
    stats = defaultdict(list)
    for s_name, split in dataset.items():
        for doc in split:
            metric_result = metric(doc)
            if isinstance(metric_result, dict):
                metric_result_flat = flatten_dict(metric_result)
                for k, v in metric_result_flat.items():
                    if isinstance(v, list):
                        stats[(s_name,) + k].extend(v)
                    else:
                        stats[(s_name,) + k].append(v)
            else:
                if isinstance(metric_result, list):
                    stats[(s_name,)].extend(metric_result)
                else:
                    stats[(s_name,)].append(metric_result)
    return dict(stats)


def test_prepare_data_simple(dataset):
    prepared_data = prepare_data(dataset=dataset, metric=lambda doc: 1)
    assert prepared_data == {
        ("train",): [1, 1, 1],
        ("test",): [1, 1, 1],
        ("validation",): [1, 1, 1],
    }
    prepared_data = prepare_data(dataset=dataset, metric=partial(label_counter, field="entities"))
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
    prepared_data = prepare_data(
        dataset=dataset, metric=partial(document_field_length_collector, field="text")
    )
    assert prepared_data == {
        ("train",): [48, 15, 19],
        ("test",): [57, 11, 40],
        ("validation",): [65, 17, 187],
    }
