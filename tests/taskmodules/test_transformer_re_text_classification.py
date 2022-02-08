import os

import pytest
import torch

from pytorch_ie.data.datasets.tacred import load_tacred
from pytorch_ie.taskmodules import TransformerRETextClassificationTaskModule
from tests import FIXTURES_ROOT


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
    taskmodule.prepare(documents)
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
        assert prepared_taskmodule.tokenizer.convert_ids_to_tokens(
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
            "take",
            "a",
            "government",
            "job",
            ".",
            "[SEP]",
        ]
    else:
        assert prepared_taskmodule.tokenizer.convert_ids_to_tokens(
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
            "take",
            "a",
            "government",
            "job",
            ".",
            "[SEP]",
        ]

    if encode_target:
        assert encoding.has_target
        target_labels = [prepared_taskmodule.id_to_label[_id] for _id in encoding.target]
        assert target_labels == ["per:title"]
    else:
        assert not encoding.has_target


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
    if prepared_taskmodule.add_type_to_marker:
        assert inputs["input_ids"].tolist() == [
            [
                101,
                1335,
                1103,
                1269,
                1159,
                117,
                2534,
                7748,
                4124,
                28997,
                4402,
                17741,
                29000,
                1209,
                1561,
                29004,
                3931,
                29007,
                117,
                13605,
                3620,
                2565,
                1150,
                1110,
                2128,
                1106,
                1321,
                170,
                1433,
                2261,
                119,
                102,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                101,
                158,
                119,
                156,
                119,
                1574,
                2031,
                5274,
                29003,
                10708,
                2061,
                29006,
                1107,
                2286,
                118,
                1428,
                3010,
                1126,
                25905,
                1222,
                160,
                12635,
                19094,
                4616,
                1170,
                1103,
                16142,
                118,
                1359,
                2950,
                28997,
                10315,
                18757,
                1200,
                29000,
                4806,
                1103,
                1751,
                1104,
                15537,
                7246,
                3300,
                1869,
                7251,
                1118,
                170,
                4267,
                1116,
                1403,
                10607,
                16271,
                1393,
                7775,
                119,
                102,
            ],
            [
                101,
                29002,
                8544,
                20595,
                1708,
                29005,
                1371,
                118,
                5004,
                118,
                5004,
                1429,
                131,
                5004,
                131,
                2724,
                11390,
                1497,
                2394,
                2206,
                2103,
                1115,
                28997,
                20018,
                13683,
                29000,
                117,
                3616,
                13606,
                117,
                1108,
                1276,
                2044,
                1118,
                1117,
                6124,
                1107,
                1103,
                24668,
                1104,
                1117,
                2123,
                3787,
                119,
                102,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
        ]
    else:
        assert inputs["input_ids"].tolist() == [
            [
                101,
                1335,
                1103,
                1269,
                1159,
                117,
                2534,
                7748,
                4124,
                28996,
                4402,
                17741,
                28997,
                1209,
                1561,
                28998,
                3931,
                28999,
                117,
                13605,
                3620,
                2565,
                1150,
                1110,
                2128,
                1106,
                1321,
                170,
                1433,
                2261,
                119,
                102,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                101,
                158,
                119,
                156,
                119,
                1574,
                2031,
                5274,
                28998,
                10708,
                2061,
                28999,
                1107,
                2286,
                118,
                1428,
                3010,
                1126,
                25905,
                1222,
                160,
                12635,
                19094,
                4616,
                1170,
                1103,
                16142,
                118,
                1359,
                2950,
                28996,
                10315,
                18757,
                1200,
                28997,
                4806,
                1103,
                1751,
                1104,
                15537,
                7246,
                3300,
                1869,
                7251,
                1118,
                170,
                4267,
                1116,
                1403,
                10607,
                16271,
                1393,
                7775,
                119,
                102,
            ],
            [
                101,
                28998,
                8544,
                20595,
                1708,
                28999,
                1371,
                118,
                5004,
                118,
                5004,
                1429,
                131,
                5004,
                131,
                2724,
                11390,
                1497,
                2394,
                2206,
                2103,
                1115,
                28996,
                20018,
                13683,
                28997,
                117,
                3616,
                13606,
                117,
                1108,
                1276,
                2044,
                1118,
                1117,
                6124,
                1107,
                1103,
                24668,
                1104,
                1117,
                2123,
                3787,
                119,
                102,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
        ]

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
    assert unbatched_output1["probabilities"] == [0.9999593496322632]

    unbatched_output2 = unbatched_outputs[1]
    assert prepared_taskmodule.label_to_id[unbatched_output2["labels"][0]] == 1
    assert unbatched_output2["probabilities"] == [0.9999768733978271]

    unbatched_output3 = unbatched_outputs[2]
    assert prepared_taskmodule.label_to_id[unbatched_output3["labels"][0]] == 2
    assert unbatched_output3["probabilities"] == [0.9999799728393555]


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
