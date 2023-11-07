import dataclasses
import json

import datasets
import pytest

from pytorch_ie.annotations import BinaryRelation, LabeledSpan, Span
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.data import Dataset, IterableDataset
from pytorch_ie.documents import TextDocument
from tests import FIXTURES_ROOT

datasets.disable_caching()


@dataclasses.dataclass
class TestDocument(TextDocument):
    sentences: AnnotationList[Span] = annotation_field(target="text")
    entities: AnnotationList[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")


@pytest.fixture
def json_dataset():
    dataset_dir = FIXTURES_ROOT / "datasets" / "json"

    dataset = datasets.load_dataset(
        path="json",
        field="data",
        data_files={
            "train": str(dataset_dir / "train.json"),
            "validation": str(dataset_dir / "val.json"),
            "test": str(dataset_dir / "test.json"),
        },
    )

    return dataset


@pytest.fixture
def iterable_json_dataset():
    dataset_dir = FIXTURES_ROOT / "datasets" / "json"

    dataset = datasets.load_dataset(
        path="json",
        field="data",
        data_files={
            "train": str(dataset_dir / "train.json"),
            "validation": str(dataset_dir / "val.json"),
            "test": str(dataset_dir / "test.json"),
        },
        streaming=True,
    )

    return dataset


def example_to_doc_dict(example):
    doc = TestDocument(text=example["text"], id=example["id"])

    doc.metadata = dict(example["metadata"])

    sentences = [Span.fromdict(dct) for dct in example["sentences"]]

    entities = [LabeledSpan.fromdict(dct) for dct in example["entities"]]

    relations = [
        BinaryRelation(head=entities[rel["head"]], tail=entities[rel["tail"]], label=rel["label"])
        for rel in example["relations"]
    ]

    for sentence in sentences:
        doc.sentences.append(sentence)

    for entity in entities:
        doc.entities.append(entity)

    for relation in relations:
        doc.relations.append(relation)

    return doc.asdict()


@pytest.fixture
def dataset(json_dataset):
    mapped_dataset = json_dataset.map(example_to_doc_dict)

    dataset = datasets.DatasetDict(
        {
            k: Dataset.from_hf_dataset(dataset, document_type=TestDocument)
            for k, dataset in mapped_dataset.items()
        }
    )

    assert len(dataset) == 3
    assert set(dataset.keys()) == {"train", "validation", "test"}

    assert len(dataset["train"]) == 8
    assert len(dataset["validation"]) == 2
    assert len(dataset["test"]) == 2

    return dataset


@pytest.fixture
def document_dataset():
    result = {}
    for path in (FIXTURES_ROOT / "datasets" / "json").iterdir():
        loaded_data = json.load(open(path, "r"))["data"]
        docs = [TestDocument.fromdict(example_to_doc_dict(ex)) for ex in loaded_data]
        result[path.stem] = docs
    return result


@pytest.fixture
def documents(document_dataset):
    return document_dataset["train"]


def test_documents(documents):
    assert len(documents) == 8
    assert all(isinstance(doc, TestDocument) for doc in documents)


@pytest.fixture
def iterable_dataset(iterable_json_dataset):
    dataset = datasets.IterableDatasetDict(
        {
            k: IterableDataset.from_hf_dataset(
                dataset.map(example_to_doc_dict), document_type=TestDocument
            )
            for k, dataset in iterable_json_dataset.items()
        }
    )

    assert len(dataset) == 3
    assert set(dataset.keys()) == {"train", "validation", "test"}

    return dataset


@pytest.fixture(params=["dataset", "iterable_dataset"])
def maybe_iterable_dataset(request):
    return request.getfixturevalue(request.param)
