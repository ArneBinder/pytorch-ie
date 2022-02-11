import copy
import os

import pytest
import torch

from pytorch_ie.data.datasets.tacred import load_tacred
from pytorch_ie.taskmodules import TransformerRETextClassificationTaskModule
from pytorch_ie.taskmodules.transformer_re_text_classification import _get_window_around_slice
from tests import FIXTURES_ROOT

TOKENS = [
    "[CLS]",
    "At",
    "the",
    "same",
    "time",
    ",",
    "Chief",
    "Financial",
    "Officer",
    "[H]",
    "Douglas",
    "Flint",
    "[/H]",
    "will",
    "become",
    "[T]",
    "chairman",
    "[/T]",
    ",",
    "succeeding",
    "Stephen",
    "Green",
    "who",
    "is",
    "leaving",
    "to",
    "take",
    "a",
    "government",
    "job",
    ".",
    "[SEP]",
]

TOKENS_WITH_MARKER = [
    "[CLS]",
    "At",
    "the",
    "same",
    "time",
    ",",
    "Chief",
    "Financial",
    "Officer",
    "[H:PERSON]",
    "Douglas",
    "Flint",
    "[/H:PERSON]",
    "will",
    "become",
    "[T:TITLE]",
    "chairman",
    "[/T:TITLE]",
    ",",
    "succeeding",
    "Stephen",
    "Green",
    "who",
    "is",
    "leaving",
    "to",
    "take",
    "a",
    "government",
    "job",
    ".",
    "[SEP]",
]


@pytest.fixture
def documents():
    documents = load_tacred(os.path.join(FIXTURES_ROOT, "datasets/tacred"), split="train")
    assert len(documents) == 3

    return documents


@pytest.fixture(scope="module", params=[False, True])
def taskmodule(request):
    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = TransformerRETextClassificationTaskModule(
        tokenizer_name_or_path=tokenizer_name_or_path, add_type_to_marker=request.param
    )
    return taskmodule


@pytest.fixture
def prepared_taskmodule(taskmodule, documents):
    taskmodule.prepare(documents)
    return taskmodule


@pytest.fixture
def model_output():
    return {
        "logits": torch.tensor(
            [
                [9.2733, -1.0300, -2.5785],
                [-1.6924, 9.5473, -1.9625],
                [-0.9995, -2.5705, 10.0095],
            ]
        ),
    }


def test_prepare(taskmodule, documents):
    assert not taskmodule.is_prepared()
    taskmodule.prepare(documents)
    assert taskmodule.is_prepared()
    assert set(taskmodule.label_to_id.keys()) == {"no_relation", "per:children", "per:title"}
    assert taskmodule.label_to_id["no_relation"] == 0
    if taskmodule.add_type_to_marker:
        assert taskmodule.argument_markers == {
            ("head", "start", "PERSON"): "[H:PERSON]",
            ("head", "start", "CITY"): "[H:CITY]",
            ("head", "start", "TITLE"): "[H:TITLE]",
            ("head", "end", "PERSON"): "[/H:PERSON]",
            ("head", "end", "CITY"): "[/H:CITY]",
            ("head", "end", "TITLE"): "[/H:TITLE]",
            ("tail", "start", "PERSON"): "[T:PERSON]",
            ("tail", "start", "CITY"): "[T:CITY]",
            ("tail", "start", "TITLE"): "[T:TITLE]",
            ("tail", "end", "PERSON"): "[/T:PERSON]",
            ("tail", "end", "CITY"): "[/T:CITY]",
            ("tail", "end", "TITLE"): "[/T:TITLE]",
        }
    else:
        assert taskmodule.argument_markers == {
            ("head", "start"): "[H]",
            ("head", "end"): "[/H]",
            ("tail", "start"): "[T]",
            ("tail", "end"): "[/T]",
        }


def test_config(prepared_taskmodule):
    config = prepared_taskmodule._config()
    assert config["taskmodule_type"] == "TransformerRETextClassificationTaskModule"
    assert "label_to_id" in config
    assert set(config["label_to_id"]) == {"no_relation", "per:children", "per:title"}


@pytest.mark.parametrize("encode_target", [False, True])
def test_encode(prepared_taskmodule, documents, encode_target):
    task_encodings = prepared_taskmodule.encode(documents, encode_target=encode_target)
    assert len(task_encodings) == 3

    encoding = task_encodings[0]
    document = documents[0]
    assert encoding.document == document
    assert "input_ids" in encoding.input
    if prepared_taskmodule.add_type_to_marker:
        assert (
            prepared_taskmodule.tokenizer.convert_ids_to_tokens(encoding.input["input_ids"])
            == TOKENS_WITH_MARKER
        )
    else:
        assert (
            prepared_taskmodule.tokenizer.convert_ids_to_tokens(encoding.input["input_ids"])
            == TOKENS
        )

    if encode_target:
        assert encoding.has_target
        target_labels = [prepared_taskmodule.id_to_label[_id] for _id in encoding.target]
        assert target_labels == ["per:title"]
    else:
        assert not encoding.has_target


def test_encode_with_windowing(prepared_taskmodule, documents):
    prepared_taskmodule_with_windowing = copy.deepcopy(prepared_taskmodule)
    prepared_taskmodule_with_windowing.max_window = 27
    task_encodings = prepared_taskmodule_with_windowing.encode(documents, encode_target=False)
    assert len(task_encodings) == 2

    encoding = task_encodings[0]
    document = documents[0]
    assert encoding.document == document
    assert "input_ids" in encoding.input
    assert len(encoding.input["input_ids"]) <= prepared_taskmodule_with_windowing.max_window
    if prepared_taskmodule_with_windowing.add_type_to_marker:
        assert prepared_taskmodule_with_windowing.tokenizer.convert_ids_to_tokens(
            encoding.input["input_ids"]
        ) == [
            "[CLS]",
            "At",
            "the",
            "same",
            "time",
            ",",
            "Chief",
            "Financial",
            "Officer",
            "[H:PERSON]",
            "Douglas",
            "Flint",
            "[/H:PERSON]",
            "will",
            "become",
            "[T:TITLE]",
            "chairman",
            "[/T:TITLE]",
            ",",
            "succeeding",
            "Stephen",
            "Green",
            "who",
            "is",
            "leaving",
            "to",
            "[SEP]",
        ]
    else:
        assert prepared_taskmodule_with_windowing.tokenizer.convert_ids_to_tokens(
            encoding.input["input_ids"]
        ) == [
            "[CLS]",
            "At",
            "the",
            "same",
            "time",
            ",",
            "Chief",
            "Financial",
            "Officer",
            "[H]",
            "Douglas",
            "Flint",
            "[/H]",
            "will",
            "become",
            "[T]",
            "chairman",
            "[/T]",
            ",",
            "succeeding",
            "Stephen",
            "Green",
            "who",
            "is",
            "leaving",
            "to",
            "[SEP]",
        ]

    encoding = task_encodings[1]
    document = documents[2]
    assert encoding.document == document
    assert "input_ids" in encoding.input
    assert len(encoding.input["input_ids"]) <= prepared_taskmodule_with_windowing.max_window
    if prepared_taskmodule_with_windowing.add_type_to_marker:
        assert prepared_taskmodule_with_windowing.tokenizer.convert_ids_to_tokens(
            encoding.input["input_ids"]
        ) == [
            "[CLS]",
            "[T:CITY]",
            "PA",
            "##RI",
            "##S",
            "[/T:CITY]",
            "2009",
            "-",
            "07",
            "-",
            "07",
            "11",
            ":",
            "07",
            ":",
            "32",
            "UTC",
            "French",
            "media",
            "earlier",
            "reported",
            "that",
            "[H:PERSON]",
            "Mont",
            "##court",
            "[/H:PERSON]",
            "[SEP]",
        ]
    else:
        assert prepared_taskmodule_with_windowing.tokenizer.convert_ids_to_tokens(
            encoding.input["input_ids"]
        ) == [
            "[CLS]",
            "[T]",
            "PA",
            "##RI",
            "##S",
            "[/T]",
            "2009",
            "-",
            "07",
            "-",
            "07",
            "11",
            ":",
            "07",
            ":",
            "32",
            "UTC",
            "French",
            "media",
            "earlier",
            "reported",
            "that",
            "[H]",
            "Mont",
            "##court",
            "[/H]",
            "[SEP]",
        ]


@pytest.mark.parametrize("encode_target", [False, True])
def test_collate(prepared_taskmodule, documents, encode_target):
    encodings = prepared_taskmodule.encode(documents, encode_target=encode_target)
    assert len(encodings) == 3

    if encode_target:
        assert all([encoding.has_target for encoding in encodings])
    else:
        assert not any([encoding.has_target for encoding in encodings])

    batch_encoding = prepared_taskmodule.collate(encodings)
    inputs, targets = batch_encoding
    assert "input_ids" in inputs
    assert "attention_mask" in inputs
    assert inputs["input_ids"].shape[0] == 3
    assert inputs["input_ids"].shape == inputs["attention_mask"].shape

    expected_tokens = TOKENS_WITH_MARKER if prepared_taskmodule.add_type_to_marker else TOKENS
    expected_token_ids = prepared_taskmodule.tokenizer.convert_tokens_to_ids(expected_tokens)
    # Note: we just add the padding tokens to the end
    n_pad = len(inputs["input_ids"].tolist()[0]) - len(expected_token_ids)
    expected_padded_input = (
        expected_token_ids + [prepared_taskmodule.tokenizer.pad_token_id] * n_pad
    )
    assert inputs["input_ids"].tolist()[0] == expected_padded_input

    if encode_target:
        assert targets.shape == (3,)
        expected_ids = [
            prepared_taskmodule.label_to_id[label]
            for label in ["per:title", "no_relation", "per:children"]
        ]
        assert targets.tolist() == expected_ids
    else:
        assert targets is None


def test_unbatch_output(prepared_taskmodule, model_output):
    unbatched_outputs = prepared_taskmodule.unbatch_output(model_output)

    assert len(unbatched_outputs) == 3

    unbatched_output1 = unbatched_outputs[0]
    assert unbatched_output1["labels"] == ["no_relation"]
    assert pytest.approx([0.9999593496322632], unbatched_output1["probabilities"])

    unbatched_output2 = unbatched_outputs[1]
    assert prepared_taskmodule.label_to_id[unbatched_output2["labels"][0]] == 1
    assert pytest.approx([0.9999768733978271], unbatched_output2["probabilities"])

    unbatched_output3 = unbatched_outputs[2]
    assert prepared_taskmodule.label_to_id[unbatched_output3["labels"][0]] == 2
    assert pytest.approx([0.9999799728393555], unbatched_output3["probabilities"])


@pytest.mark.parametrize("inplace", [True, False])
def test_decode(prepared_taskmodule, documents, model_output, inplace):
    encodings = prepared_taskmodule.encode(documents, encode_target=False)
    unbatched_outputs = prepared_taskmodule.unbatch_output(model_output)
    decoded_documents = prepared_taskmodule.decode(
        encodings=encodings, decoded_outputs=unbatched_outputs, inplace=inplace
    )

    assert len(decoded_documents) == len(documents)
    if inplace:
        assert set(decoded_documents) == set(documents)
    else:
        assert set(decoded_documents).isdisjoint(set(documents))

    # sort documents because order of documents is not deterministic if inplace==False
    decoded_documents = sorted(decoded_documents, key=lambda doc: doc.text)
    decoded_document = decoded_documents[1]
    predictions = decoded_document.predictions("relations")
    assert len(predictions) == 1
    assert prepared_taskmodule.label_to_id[predictions[0].label] == 2
    head = predictions[0].head
    assert head.label == "PERSON"
    assert head.start == 65
    assert head.end == 74
    tail = predictions[0].tail
    assert tail.label == "CITY"
    assert tail.start == 0
    assert tail.end == 5


def test_save_load(tmp_path, prepared_taskmodule):
    path = os.path.join(tmp_path, "taskmodule")
    prepared_taskmodule.save_pretrained(path)
    loaded_taskmodule = TransformerRETextClassificationTaskModule.from_pretrained(path)
    assert loaded_taskmodule.is_prepared()
    assert loaded_taskmodule.argument_markers == prepared_taskmodule.argument_markers


def test_get_window_around_slice():

    # default: result is centered around slice
    window_slice = _get_window_around_slice(
        slice=(5, 7), max_window_size=6, available_input_length=10
    )
    assert window_slice == (3, 9)

    # slice at the beginning -> shift window to the right (regarding the slice center)
    window_slice = _get_window_around_slice(
        slice=(0, 5), max_window_size=8, available_input_length=10
    )
    assert window_slice == (0, 8)

    # slice at the end -> shift window to the left (regarding the slice center)
    window_slice = _get_window_around_slice(
        slice=(7, 10), max_window_size=8, available_input_length=10
    )
    assert window_slice == (2, 10)

    # max window size bigger than available_input_length -> take everything
    window_slice = _get_window_around_slice(
        slice=(2, 6), max_window_size=8, available_input_length=7
    )
    assert window_slice == (0, 7)

    # slice exceeds max_window_size
    window_slice = _get_window_around_slice(
        slice=(0, 5), max_window_size=4, available_input_length=10
    )
    assert window_slice is None
