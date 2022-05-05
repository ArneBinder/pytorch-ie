import numpy
import pytest
import torch

from pytorch_ie.core import TaskModule
from pytorch_ie.taskmodules.transformer_span_classification import (
    TransformerSpanClassificationTaskModule,
)


@pytest.fixture(scope="module")
def taskmodule():
    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = TransformerSpanClassificationTaskModule(
        tokenizer_name_or_path=tokenizer_name_or_path
    )
    return taskmodule


@pytest.fixture
def prepared_taskmodule(taskmodule, documents):
    taskmodule.prepare(documents)
    return taskmodule


@pytest.fixture
def model_output():
    return {
        "logits": torch.from_numpy(
            numpy.log(
                [
                    # O, ORG, PER
                    [0.5, 0.2, 0.3],
                    [0.1, 0.1, 0.8],
                    [0.1, 0.5, 0.4],
                    [0.1, 0.4, 0.5],
                    [0.1, 0.6, 0.3],
                ]
            )
        ),
        "start_indices": torch.tensor([1, 1, 7, 1, 6]),
        "end_indices": torch.tensor([2, 4, 7, 4, 6]),
        "batch_indices": torch.tensor([0, 1, 1, 2, 2]),
    }


def test_prepare(taskmodule, documents):
    taskmodule.prepare(documents)
    assert set(taskmodule.label_to_id.keys()) == {"PER", "ORG", "O"}
    assert [taskmodule.id_to_label[i] for i in range(3)] == ["O", "ORG", "PER"]
    assert taskmodule.label_to_id["O"] == 0


def test_config(prepared_taskmodule):
    config = prepared_taskmodule._config()
    assert config["taskmodule_type"] == "TransformerSpanClassificationTaskModule"
    assert "label_to_id" in config
    assert config["label_to_id"] == {"O": 0, "ORG": 1, "PER": 2}


@pytest.mark.parametrize("encode_target", [False, True])
def test_encode(prepared_taskmodule, documents, encode_target):
    task_encodings = prepared_taskmodule.encode(documents, encode_target=encode_target)
    assert len(task_encodings) == 8

    task_encoding = task_encodings[5]
    document = documents[5]
    assert task_encoding.document == document
    assert "input_ids" in task_encoding.inputs
    assert (
        prepared_taskmodule.tokenizer.decode(
            task_encoding.inputs["input_ids"], skip_special_tokens=True
        )
        == document.text
    )

    if encode_target:
        assert task_encoding.targets == [
            (1, 4, prepared_taskmodule.label_to_id["PER"]),
            (6, 6, prepared_taskmodule.label_to_id["ORG"]),
            (9, 9, prepared_taskmodule.label_to_id["ORG"]),
        ]
    else:
        assert not task_encoding.has_targets


def test_unbatch_output(prepared_taskmodule, model_output):
    unbatched_outputs = prepared_taskmodule.unbatch_output(model_output)

    assert len(unbatched_outputs) == 3

    assert unbatched_outputs[0] == {"tags": [], "probabilities": []}

    assert unbatched_outputs[1] == {
        "tags": [("PER", (1, 4)), ("ORG", (7, 7))],
        "probabilities": pytest.approx([0.8, 0.5]),
    }

    assert unbatched_outputs[2] == {
        "tags": [("PER", (1, 4)), ("ORG", (6, 6))],
        "probabilities": pytest.approx([0.5, 0.6]),
    }


@pytest.mark.parametrize("inplace", [False, True])
def test_decode(prepared_taskmodule, documents, model_output, inplace):
    documents = documents[:3]

    encodings = prepared_taskmodule.encode(documents, encode_target=False)
    unbatched_outputs = prepared_taskmodule.unbatch_output(model_output)
    decoded_documents = prepared_taskmodule.decode(
        task_encodings=encodings,
        task_outputs=unbatched_outputs,
        inplace=inplace,
    )

    assert len(decoded_documents) == len(documents)

    if inplace:
        assert {id(doc) for doc in decoded_documents} == {id(doc) for doc in documents}
    else:
        assert {id(doc) for doc in decoded_documents}.isdisjoint({id(doc) for doc in documents})

    expected_scores = [0.8, 0.5, 0.5, 0.6]
    i = 0
    for document in decoded_documents:
        for entity_expected, entity_decoded in zip(
            document["entities"], document["entities"].predictions
        ):
            assert entity_expected.start == entity_decoded.start
            assert entity_expected.end == entity_decoded.end
            assert entity_expected.label == entity_decoded.label
            assert expected_scores[i] == pytest.approx(entity_decoded.score)
            i += 1

    if not inplace:
        for document in documents:
            assert not document["entities"].predictions


@pytest.mark.parametrize("encode_target", [False, True])
def test_collate(prepared_taskmodule, documents, encode_target):
    documents = documents[:3]

    task_encodings = prepared_taskmodule.encode(documents, encode_target=encode_target)
    assert len(task_encodings) == 3

    if encode_target:
        assert all([task_encoding.has_targets for task_encoding in task_encodings])
    else:
        assert all([not task_encoding.has_targets for task_encoding in task_encodings])

    batch_encoding = prepared_taskmodule.collate(task_encodings)
    inputs, targets = batch_encoding
    assert inputs["input_ids"].shape[0] == 3

    if encode_target:
        assert len(targets) == 3
    else:
        assert targets is None


def test_load_from_registry():
    taskmodule_type = TaskModule.by_name("TransformerSpanClassificationTaskModule")
    assert taskmodule_type is TransformerSpanClassificationTaskModule

    tokenizer_name_or_path = "bert-base-cased"
    taskmodule_type(tokenizer_name_or_path=tokenizer_name_or_path)
