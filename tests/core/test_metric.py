from dataclasses import dataclass
from typing import Optional

import pytest

from pytorch_ie.annotations import LabeledSpan
from pytorch_ie.core import AnnotationList, Document, DocumentMetric, annotation_field
from pytorch_ie.core.metric import T
from pytorch_ie.documents import TextBasedDocument


@pytest.fixture
def documents():
    @dataclass
    class TextDocumentWithEntities(TextBasedDocument):
        entities: AnnotationList[LabeledSpan] = annotation_field(target="text")

    # a test sentence with two entities
    doc1 = TextDocumentWithEntities(
        text="The quick brown fox jumps over the lazy dog.",
    )
    doc1.entities.append(LabeledSpan(start=4, end=19, label="animal"))
    doc1.entities.append(LabeledSpan(start=35, end=43, label="animal"))
    assert str(doc1.entities[0]) == "quick brown fox"
    assert str(doc1.entities[1]) == "lazy dog"

    # a second test sentence with a different text and a single entity (a company)
    doc2 = TextDocumentWithEntities(text="Apple is a great company.")
    doc2.entities.append(LabeledSpan(start=0, end=5, label="company"))
    assert str(doc2.entities[0]) == "Apple"

    documents = [doc1, doc2]

    # add predictions
    # correct
    documents[0].entities.predictions.append(LabeledSpan(start=4, end=19, label="animal"))
    # correct, but duplicate, this should not be counted
    documents[0].entities.predictions.append(LabeledSpan(start=4, end=19, label="animal"))
    # correct
    documents[0].entities.predictions.append(LabeledSpan(start=35, end=43, label="animal"))
    # wrong label
    documents[0].entities.predictions.append(LabeledSpan(start=35, end=43, label="cat"))
    # correct
    documents[1].entities.predictions.append(LabeledSpan(start=0, end=5, label="company"))
    # wrong span
    documents[1].entities.predictions.append(LabeledSpan(start=10, end=15, label="company"))

    return documents


class Accuracy(DocumentMetric):
    def __init__(self, layer: str):
        super().__init__()
        self.layer = layer

    def reset(self) -> None:
        self.total = 0
        self.correct = 0

    def _update(self, document: Document) -> None:
        layer = document[self.layer]
        predictions = layer.predictions
        self.total += len(set(predictions))
        self.correct += len(set(layer) & set(predictions))

    def _compute(self) -> Optional[float]:
        if self.total == 0:
            return None
        return self.correct / self.total


def test_document_metric(documents):
    accuracy = Accuracy(layer="entities")
    accuracy(documents[0])
    assert accuracy.total == 3
    assert accuracy.correct == 2
    assert accuracy.compute() == 2 / 3
    assert accuracy.total == 0
    assert accuracy.correct == 0


def test_document_metric_iterable(documents):
    accuracy = Accuracy(layer="entities")
    accuracy(documents)
    assert accuracy.total == 5
    assert accuracy.correct == 3
    assert accuracy.compute() == 3 / 5
    assert accuracy.total == 0
    assert accuracy.correct == 0


def test_document_metric_wrong_iterable():
    accuracy = Accuracy(layer="entities")
    with pytest.raises(TypeError) as excinfo:
        accuracy([1, 2])
    assert (
        str(excinfo.value)
        == "document_or_collection contains an object that is not a document: <class 'int'>"
    )


def test_document_metric_dict(documents):
    dummy_dataset_dict = {"train": [documents[0]], "val": [], "test": [documents[1]]}
    accuracy = Accuracy(layer="entities")
    result = accuracy(dummy_dataset_dict)

    assert result["train"] == 2 / 3
    assert result["test"] == 0.5
    assert result["val"] is None


def test_document_metric_wrong_type():
    accuracy = Accuracy(layer="entities")
    with pytest.raises(TypeError) as excinfo:
        accuracy(1)
    assert str(excinfo.value) == "document_or_collection has unknown type: <class 'int'>"
