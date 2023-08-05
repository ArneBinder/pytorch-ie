from dataclasses import dataclass

import pytest

from pytorch_ie.annotations import LabeledSpan
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.documents import TextBasedDocument
from pytorch_ie.metrics import F1Metric


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


def test_f1(documents):
    metric = F1Metric(layer="entities")
    metric(documents)
    # tp, fp, fn for micro
    assert dict(metric.counts) == {"MICRO": (3, 2, 0)}
    assert metric.compute() == {"MICRO": {"f1": 0.7499999999999999, "p": 0.6, "r": 1.0}}


def test_f1_per_label(documents):
    metric = F1Metric(layer="entities", labels=["animal", "company", "cat"])
    metric(documents)
    # tp, fp, fn for micro and per label
    assert dict(metric.counts) == {
        "MICRO": (3, 2, 0),
        "cat": (0, 1, 0),
        "company": (1, 1, 0),
        "animal": (2, 0, 0),
    }
    assert metric.compute() == {
        "MACRO": {"f1": 0.5555555555555556, "p": 0.5, "r": 0.6666666666666666},
        "MICRO": {"f1": 0.7499999999999999, "p": 0.6, "r": 1.0},
        "cat": {"f1": 0.0, "p": 0.0, "r": 0.0},
        "company": {"f1": 0.6666666666666666, "p": 0.5, "r": 1.0},
        "animal": {"f1": 1.0, "p": 1.0, "r": 1.0},
    }


def test_f1_per_label_no_labels(documents):
    with pytest.raises(ValueError) as excinfo:
        F1Metric(layer="entities", labels=[])
    assert str(excinfo.value) == "labels cannot be empty"


def test_f1_per_label_not_allowed():
    with pytest.raises(ValueError) as excinfo:
        F1Metric(layer="entities", labels=["animal", "MICRO"])
    assert (
        str(excinfo.value)
        == "labels cannot contain 'MICRO' or 'MACRO' because they are used to capture aggregated metrics"
    )

    assert result == {
        "train": {"MICRO": {"f1": 0.8, "p": 0.6666666666666666, "r": 1.0}},
        "val": {},
        "test": {"MICRO": {"f1": 0.6666666666666666, "p": 0.5, "r": 1.0}},
    }
