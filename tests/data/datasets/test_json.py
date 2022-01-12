import os

from pytorch_ie.data.datasets.json import load_json
from tests import FIXTURES_ROOT


def test_load_json():
    documents = load_json(os.path.join(FIXTURES_ROOT, "datasets/json/train.json"))
    assert len(documents) == 3

    document = documents[2]
    assert document.text == "Jim works at Hamburg University."

    entities = document.span_annotations("entities")
    assert len(entities) == 2

    entity = entities[1]
    assert entity.start == 13
    assert entity.end == 31
    assert entity.label == "ORG"
