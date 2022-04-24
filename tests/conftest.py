import dataclasses

import datasets
import pytest

from pytorch_ie.annotations import BinaryRelation, LabeledSpan, Span
from pytorch_ie.document import TextDocument, annotation_field, AnnotationList
from pytorch_ie.data import Dataset
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
def dataset(json_dataset):
    def example_to_doc_dict(example):
        doc = TestDocument(text=example["text"], id=example["id"])

        doc.metadata = dict(example["metadata"])

        sentences = [Span.fromdict(dct) for dct in example["sentences"]]

        entities = [LabeledSpan.fromdict(dct) for dct in example["entities"]]

        relations = [
            BinaryRelation(
                head=entities[rel["head"]], tail=entities[rel["tail"]], label=rel["label"]
            )
            for rel in example["relations"]
        ]

        for sentence in sentences:
            doc.sentences.append(sentence)

        for entity in entities:
            doc.entities.append(entity)

        for relation in relations:
            doc.relations.append(relation)

        return doc.asdict()

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
def documents(dataset):
    return list(dataset["train"])
