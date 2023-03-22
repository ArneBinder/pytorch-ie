import json
import pickle as pkl
from typing import Type, TypeVar

import pytest

from pytorch_ie.core import Document
from pytorch_ie.taskmodules.multi_modal_sequence_tagging import (
    MultiModalSequenceTaggingTaskModule,
    OcrDocumentWithEntities,
)
from tests import FIXTURES_ROOT

T_doc = TypeVar("T_doc", bound=Document)


def _load_doc_from_json(path: str, document_type: Type[T_doc]) -> T_doc:
    doc_json = json.load(open(path))
    doc = document_type.fromdict(doc_json)
    return doc


@pytest.fixture(scope="module")
def document_with_entities():
    return _load_doc_from_json(
        path=FIXTURES_ROOT / "datasets" / "xfund" / "train_0_converted.json",
        document_type=OcrDocumentWithEntities,
    )


@pytest.fixture(scope="module")
def dataset(document_with_entities):
    return [document_with_entities]


def test_dataset(dataset):
    pass


@pytest.fixture(scope="module")
def taskmodule():
    tm = MultiModalSequenceTaggingTaskModule(
        processor_name_or_path="microsoft/layoutxlm-base",
        processor_kwargs=dict(apply_ocr=False),
        exclude_labels=["other"],
        # use a lower max_length as possible (would be 512) to get multiple batch entries
        max_length=128,
    )
    return tm


def test_taskmodule(taskmodule):
    pass


@pytest.fixture(scope="module")
def prepared_taskmodule(taskmodule, dataset):
    assert len(taskmodule._config()["label_to_id"]) == 0
    taskmodule.prepare(documents=dataset)
    return taskmodule


def test_prepared_taskmodule(prepared_taskmodule):
    assert prepared_taskmodule._config()["label_to_id"] == {
        "O": 0,
        "B-answer": 1,
        "B-header": 3,
        "B-question": 5,
        "I-answer": 2,
        "I-header": 4,
        "I-question": 6,
    }


@pytest.fixture(scope="module")
def task_encodings(prepared_taskmodule, dataset):
    res = [prepared_taskmodule.encode_input(document) for document in dataset]
    return res


def test_encode_input(task_encodings):
    assert task_encodings is not None


@pytest.fixture(scope="module")
def task_encodings_with_targets(prepared_taskmodule, dataset):
    return prepared_taskmodule.encode(dataset, encode_target=True)


@pytest.fixture(scope="module")
def batches(prepared_taskmodule, task_encodings_with_targets):
    batch_size = 4
    _batches = []
    for idx in range(0, len(task_encodings_with_targets), batch_size):
        batch_features = task_encodings_with_targets[idx : idx + batch_size]
        batch = prepared_taskmodule.collate(batch_features)
        assert batch is not None
        _batches.append(batch)
    return _batches


def test_collate(batches):
    assert batches is not None


@pytest.fixture(scope="module", params=[0, 1])
def model_output(request):
    fn = FIXTURES_ROOT / "model_output" / f"layoutxml_xfund_batch_{request.param}.pkl"
    return pkl.load(open(fn, "rb"))


def test_model_output(model_output):
    assert model_output is not None
    # assert model_output.loss is None
    assert model_output.logits is not None


@pytest.fixture(scope="module")
def unbatched_model_output(prepared_taskmodule, model_output):
    return prepared_taskmodule.unbatch_output(model_output)


def test_unbatch_output(unbatched_model_output):
    assert unbatched_model_output is not None


@pytest.fixture(scope="module")
def annotations_from_output(
    prepared_taskmodule, task_encodings_with_targets, unbatched_model_output
):
    return list(
        prepared_taskmodule.create_annotations_from_output(
            task_encodings=task_encodings_with_targets[0], task_outputs=unbatched_model_output[0]
        )
    )


def test_create_annotations_from_output(annotations_from_output):
    assert annotations_from_output is not None
