import dataclasses
import json

import pytest

from pytorch_ie.annotations import BinaryRelation, LabeledSpan, Span
from pytorch_ie.core import AnnotationLayer, annotation_field
from pytorch_ie.documents import TextDocument
from tests import FIXTURES_ROOT


@dataclasses.dataclass
class TestDocument(TextDocument):
    sentences: AnnotationLayer[Span] = annotation_field(target="text")
    entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationLayer[BinaryRelation] = annotation_field(target="entities")


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
def document_dataset():
    result = {}
    for path in (FIXTURES_ROOT / "datasets" / "json").iterdir():
        loaded_data = json.load(open(path))["data"]
        docs = [TestDocument.fromdict(example_to_doc_dict(ex)) for ex in loaded_data]
        result[path.stem] = docs
    return result


@pytest.fixture
def documents(document_dataset):
    return document_dataset["train"]


def test_documents(documents):
    assert len(documents) == 8
    assert all(isinstance(doc, TestDocument) for doc in documents)
