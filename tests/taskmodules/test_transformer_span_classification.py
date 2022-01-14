import numpy
import pytest
import torch

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
            numpy.log([[0.2, 0.5, 0.3], [0.8, 0.1, 0.1], [0.1, 0.4, 0.5], [0.1, 0.5, 0.4]])
        ),
        "start_indices": torch.tensor([1, 2, 3, 4]),
        "end_indices": torch.tensor([2, 3, 4, 5]),
        "batch_indices": torch.tensor([0, 1, 2, 2]),
    }


def test_prepare(taskmodule, documents):
    taskmodule.prepare(documents)
    assert set(taskmodule.label_to_id.keys()) == {"PER", "LOC", "ORG", "O"}
    assert taskmodule.label_to_id["O"] == 0


def test_config(prepared_taskmodule):
    config = prepared_taskmodule._config()
    assert config["taskmodule_type"] == "TransformerSpanClassificationTaskModule"
    assert "label_to_id" in config
    assert set(config["label_to_id"]) == {"PER", "LOC", "ORG", "O"}


@pytest.mark.parametrize("encode_target", [False, True])
def test_encode_without_target(prepared_taskmodule, documents, encode_target):
    task_encodings = prepared_taskmodule.encode(documents, encode_target=encode_target)
    assert len(task_encodings) == 3

    encoding = task_encodings[0]
    document = documents[0]
    assert encoding.document == document
    assert "input_ids" in encoding.input
    assert (
        prepared_taskmodule.tokenizer.decode(encoding.input["input_ids"], skip_special_tokens=True)
        == document.text
    )

    if encode_target:
        assert encoding.target == [
            (1, 1, prepared_taskmodule.label_to_id["PER"]),
            (4, 4, prepared_taskmodule.label_to_id["LOC"]),
            (6, 6, prepared_taskmodule.label_to_id["LOC"]),
            (9, 9, prepared_taskmodule.label_to_id["LOC"]),
        ]
    else:
        assert encoding.target is None


def test_unbatch_output(prepared_taskmodule, model_output):
    unbatched_outputs = prepared_taskmodule.unbatch_output(model_output)

    assert len(unbatched_outputs) == 3

    unbatched_output1 = unbatched_outputs[0]
    assert unbatched_output1["tags"] == [(prepared_taskmodule.id_to_label[1], (1, 2))]
    assert len(unbatched_output1["probabilities"]) == 1
    assert pytest.approx(unbatched_output1["probabilities"][0], 0.5)

    unbatched_output2 = unbatched_outputs[1]
    assert len(unbatched_output2["tags"]) == 0
    assert len(unbatched_output2["probabilities"]) == 0

    unbatched_output3 = unbatched_outputs[2]
    assert len(unbatched_output3["tags"]) == 2
    assert len(unbatched_output3["probabilities"]) == 2


def test_decode_not_inplace(prepared_taskmodule, documents, model_output):
    encodings = prepared_taskmodule.encode(documents, encode_target=False)
    unbatched_outputs = prepared_taskmodule.unbatch_output(model_output)
    decoded_documents = prepared_taskmodule.decode(
        encodings=encodings, decoded_outputs=unbatched_outputs, inplace=False
    )

    assert len(decoded_documents) == len(documents)
    assert set(decoded_documents).isdisjoint(set(documents))

    decoded_document = decoded_documents[2]
    predictions = decoded_document.predictions("entities")
    assert len(predictions) == 2
    assert predictions[0].start == 10
    assert predictions[0].end == 20


def test_decode_inplace(prepared_taskmodule, documents, model_output):
    encodings = prepared_taskmodule.encode(documents, encode_target=False)
    unbatched_outputs = prepared_taskmodule.unbatch_output(model_output)
    decoded_documents = prepared_taskmodule.decode(
        encodings=encodings, decoded_outputs=unbatched_outputs, inplace=True
    )

    assert len(decoded_documents) == len(documents)
    assert set(decoded_documents) == set(documents)

    decoded_document = decoded_documents[2]
    predictions = decoded_document.predictions("entities")
    assert len(predictions) == 2
    assert predictions[0].start == 10
    assert predictions[0].end == 20


@pytest.mark.parametrize("encode_target", [False, True])
def test_collate(prepared_taskmodule, documents, encode_target):
    encodings = prepared_taskmodule.encode(documents, encode_target=encode_target)
    assert len(encodings) == 3

    if encode_target:
        assert all([encoding.target is not None for encoding in encodings])
    else:
        assert all([encoding.target is None for encoding in encodings])

    batch_encoding = prepared_taskmodule.collate(encodings)
    inputs, targets = batch_encoding
    assert inputs["input_ids"].shape[0] == 3

    if encode_target:
        assert len(targets) == 3
    else:
        assert targets is None
