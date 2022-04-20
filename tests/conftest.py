import dataclasses

import datasets
import pytest

from pytorch_ie.annotations import AnnotationList, BinaryRelation, LabeledSpan, Span
from pytorch_ie.data import Dataset
from pytorch_ie.document import TextDocument, annotation_field
from tests import FIXTURES_ROOT

datasets.set_caching_enabled(False)


@dataclasses.dataclass
class TestDocument(TextDocument):
    sentences: AnnotationList[Span] = annotation_field(target="text")
    entities: AnnotationList[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")


@pytest.fixture
def dataset():
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
def document_dataset(dataset):
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

    mapped_dataset = dataset.map(example_to_doc_dict)

    doc_dataset = datasets.DatasetDict(
        {
            k: Dataset.from_hf_dataset(dataset, document_type=TestDocument)
            for k, dataset in mapped_dataset.items()
        }
    )

    assert len(doc_dataset) == 3
    assert set(doc_dataset.keys()) == {"train", "validation", "test"}

    assert len(doc_dataset["train"]) == 8
    assert len(doc_dataset["validation"]) == 2
    assert len(doc_dataset["test"]) == 2

    return doc_dataset
