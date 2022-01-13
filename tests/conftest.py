import os

import pytest

from pytorch_ie.data.datasets.json import load_json
from tests import FIXTURES_ROOT


@pytest.fixture
def documents():
    documents = load_json(os.path.join(FIXTURES_ROOT, "datasets/json/train.json"))
    assert len(documents) == 3

    return documents
