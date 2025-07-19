from dataclasses import dataclass

import pytest
from pie_core import AnnotationLayer, annotation_field

from pytorch_ie.annotations import LabeledSpan
from pytorch_ie.documents import TextBasedDocument
from pytorch_ie.metrics import ConfusionMatrix


@pytest.fixture
def documents():
    @dataclass
    class TextDocumentWithEntities(TextBasedDocument):
        entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")

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
    # wrong label
    documents[0].entities.predictions.append(LabeledSpan(start=35, end=43, label="cat"))
    # correct
    documents[1].entities.predictions.append(LabeledSpan(start=0, end=5, label="company"))
    # wrong span
    documents[1].entities.predictions.append(LabeledSpan(start=10, end=15, label="company"))

    return documents


def test_confusion_matrix(documents):
    metric = ConfusionMatrix(layer="entities")
    metric(documents)
    # (gold_label, predicted_label): count
    assert dict(metric.counts) == {
        ("animal", "animal"): 1,
        ("animal", "cat"): 1,
        ("UNASSIGNABLE", "company"): 1,
        ("company", "company"): 1,
    }
    assert metric.compute() == {
        "animal": {"animal": 1, "cat": 1},
        "UNASSIGNABLE": {"company": 1},
        "company": {"company": 1},
    }
