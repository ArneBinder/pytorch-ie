import re
from typing import Any, Dict

import numpy
import pytest
import torch

from pytorch_ie.taskmodules import TransformerRETextClassificationTaskModule


def _config_to_str(cfg: Dict[str, Any]) -> str:
    result = "-".join([f"{k}={cfg[k]}" for k in sorted(cfg)])
    return result


CONFIGS = [
    {"add_type_to_marker": False, "append_markers": False},
    {"add_type_to_marker": True, "append_markers": False},
    {"add_type_to_marker": False, "append_markers": True},
    {"add_type_to_marker": True, "append_markers": True},
]
CONFIGS_DICT = {_config_to_str(cfg): cfg for cfg in CONFIGS}


@pytest.fixture(scope="module", params=CONFIGS_DICT.keys())
def cfg(request):
    return CONFIGS_DICT[request.param]


@pytest.fixture(scope="module")
def taskmodule(cfg):
    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = TransformerRETextClassificationTaskModule(
        tokenizer_name_or_path=tokenizer_name_or_path, relation_annotation="relations", **cfg
    )
    assert not taskmodule.is_from_pretrained

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
                    # O, org:founded_by, per:employee_of, per:founder
                    [0.1, 0.6, 0.1, 0.2],
                    [0.5, 0.2, 0.2, 0.1],
                    [0.1, 0.2, 0.6, 0.1],
                    [0.1, 0.2, 0.2, 0.5],
                    [0.2, 0.4, 0.3, 0.1],
                    [0.5, 0.2, 0.2, 0.1],
                    [0.6, 0.1, 0.2, 0.1],
                    [0.5, 0.2, 0.2, 0.1],
                ]
            )
        ),
    }


def test_prepare(taskmodule, documents):
    assert not taskmodule.is_prepared
    taskmodule.prepare(documents)
    assert taskmodule.is_prepared

    assert taskmodule.entity_labels == ["ORG", "PER"]
    assert taskmodule.sep_token_id

    if taskmodule.append_markers:
        if taskmodule.add_type_to_marker:
            assert taskmodule.argument_markers == [
                "[/H:ORG]",
                "[/H:PER]",
                "[/H]",
                "[/T:ORG]",
                "[/T:PER]",
                "[/T]",
                "[H:ORG]",
                "[H:PER]",
                "[H=ORG]",
                "[H=PER]",
                "[H]",
                "[T:ORG]",
                "[T:PER]",
                "[T=ORG]",
                "[T=PER]",
                "[T]",
            ]
            assert taskmodule.argument_markers_to_id == {
                "[/H:ORG]": 28996,
                "[/H:PER]": 28997,
                "[/H]": 28998,
                "[/T:ORG]": 28999,
                "[/T:PER]": 29000,
                "[/T]": 29001,
                "[H:ORG]": 29002,
                "[H:PER]": 29003,
                "[H=ORG]": 29004,
                "[H=PER]": 29005,
                "[H]": 29006,
                "[T:ORG]": 29007,
                "[T:PER]": 29008,
                "[T=ORG]": 29009,
                "[T=PER]": 29010,
                "[T]": 29011,
            }

        else:
            assert taskmodule.argument_markers == [
                "[/H]",
                "[/T]",
                "[H=ORG]",
                "[H=PER]",
                "[H]",
                "[T=ORG]",
                "[T=PER]",
                "[T]",
            ]
            assert taskmodule.argument_markers_to_id == {
                "[/H]": 28996,
                "[/T]": 28997,
                "[H=ORG]": 28998,
                "[H=PER]": 28999,
                "[H]": 29000,
                "[T=ORG]": 29001,
                "[T=PER]": 29002,
                "[T]": 29003,
            }
    else:
        if taskmodule.add_type_to_marker:
            assert taskmodule.argument_markers == [
                "[/H:ORG]",
                "[/H:PER]",
                "[/H]",
                "[/T:ORG]",
                "[/T:PER]",
                "[/T]",
                "[H:ORG]",
                "[H:PER]",
                "[H]",
                "[T:ORG]",
                "[T:PER]",
                "[T]",
            ]
            assert taskmodule.argument_markers_to_id == {
                "[/H:ORG]": 28996,
                "[/H:PER]": 28997,
                "[/H]": 28998,
                "[/T:ORG]": 28999,
                "[/T:PER]": 29000,
                "[/T]": 29001,
                "[H:ORG]": 29002,
                "[H:PER]": 29003,
                "[H]": 29004,
                "[T:ORG]": 29005,
                "[T:PER]": 29006,
                "[T]": 29007,
            }
        else:
            assert taskmodule.argument_markers == ["[/H]", "[/T]", "[H]", "[T]"]
            assert taskmodule.argument_markers_to_id == {
                "[/H]": 28996,
                "[/T]": 28997,
                "[H]": 28998,
                "[T]": 28999,
            }

    assert taskmodule.label_to_id == {
        "org:founded_by": 1,
        "per:employee_of": 2,
        "per:founder": 3,
        "no_relation": 0,
    }
    assert taskmodule.id_to_label == {
        1: "org:founded_by",
        2: "per:employee_of",
        3: "per:founder",
        0: "no_relation",
    }


def test_config(prepared_taskmodule):
    config = prepared_taskmodule._config()
    assert config["taskmodule_type"] == "TransformerRETextClassificationTaskModule"
    assert "label_to_id" in config
    assert config["label_to_id"] == {
        "org:founded_by": 1,
        "per:employee_of": 2,
        "per:founder": 3,
        "no_relation": 0,
    }

    assert config["entity_labels"] == ["ORG", "PER"]


@pytest.mark.parametrize("encode_target", [False, True])
def test_encode(prepared_taskmodule, documents, encode_target):
    task_encodings = prepared_taskmodule.encode(documents, encode_target=encode_target)

    assert len(task_encodings) == 7

    encoding = task_encodings[0]

    tokens = prepared_taskmodule.tokenizer.convert_ids_to_tokens(encoding.inputs["input_ids"])
    assert len(tokens) == len(encoding.inputs["input_ids"])

    if prepared_taskmodule.add_type_to_marker:
        assert tokens[:14] == [
            "[CLS]",
            "[H:PER]",
            "En",
            "##ti",
            "##ty",
            "A",
            "[/H:PER]",
            "works",
            "at",
            "[T:ORG]",
            "B",
            "[/T:ORG]",
            ".",
            "[SEP]",
        ]
    else:
        assert tokens[:14] == [
            "[CLS]",
            "[H]",
            "En",
            "##ti",
            "##ty",
            "A",
            "[/H]",
            "works",
            "at",
            "[T]",
            "B",
            "[/T]",
            ".",
            "[SEP]",
        ]
    if prepared_taskmodule.append_markers:
        assert len(tokens) == 14 + 4
        assert tokens[-4:] == ["[H=PER]", "[SEP]", "[T=ORG]", "[SEP]"]
    else:
        assert len(tokens) == 14

    if encode_target:
        assert encoding.targets == [2]
    else:
        assert not encoding.has_targets

        with pytest.raises(AssertionError, match=re.escape("task encoding has no target")):
            encoding.targets


@pytest.mark.parametrize("encode_target", [False, True])
def test_collate(prepared_taskmodule, documents, encode_target):
    documents = [documents[i] for i in [0, 1, 4]]

    encodings = prepared_taskmodule.encode(documents, encode_target=encode_target)

    assert len(encodings) == 4
    if encode_target:
        assert all([encoding.has_targets for encoding in encodings])
    else:
        assert not any([encoding.has_targets for encoding in encodings])

    batch_encoding = prepared_taskmodule.collate(encodings[:2])
    inputs, targets = batch_encoding

    assert "input_ids" in inputs
    assert "attention_mask" in inputs
    assert inputs["input_ids"].shape == inputs["attention_mask"].shape

    if prepared_taskmodule.append_markers:
        assert inputs["input_ids"].shape == (2, 25)
        if prepared_taskmodule.add_type_to_marker:
            torch.testing.assert_close(
                inputs.input_ids,
                torch.tensor(
                    [
                        [
                            101,
                            29003,
                            13832,
                            3121,
                            2340,
                            138,
                            28997,
                            1759,
                            1120,
                            29007,
                            139,
                            28999,
                            119,
                            102,
                            29005,
                            102,
                            29009,
                            102,
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
                            1752,
                            5650,
                            119,
                            29003,
                            13832,
                            3121,
                            2340,
                            144,
                            28997,
                            1759,
                            1120,
                            29007,
                            145,
                            28999,
                            119,
                            1262,
                            1771,
                            146,
                            119,
                            102,
                            29005,
                            102,
                            29009,
                            102,
                        ],
                    ]
                ),
            )
        else:
            torch.testing.assert_close(
                inputs.input_ids,
                torch.tensor(
                    [
                        [
                            101,
                            29000,
                            13832,
                            3121,
                            2340,
                            138,
                            28996,
                            1759,
                            1120,
                            29003,
                            139,
                            28997,
                            119,
                            102,
                            28999,
                            102,
                            29001,
                            102,
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
                            1752,
                            5650,
                            119,
                            29000,
                            13832,
                            3121,
                            2340,
                            144,
                            28996,
                            1759,
                            1120,
                            29003,
                            145,
                            28997,
                            119,
                            1262,
                            1771,
                            146,
                            119,
                            102,
                            28999,
                            102,
                            29001,
                            102,
                        ],
                    ]
                ),
            )
        torch.testing.assert_close(
            inputs.attention_mask,
            torch.tensor(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ]
            ),
        )

    else:
        assert inputs["input_ids"].shape == (2, 21)

        if prepared_taskmodule.add_type_to_marker:
            torch.testing.assert_close(
                inputs.input_ids,
                torch.tensor(
                    [
                        [
                            101,
                            29003,
                            13832,
                            3121,
                            2340,
                            138,
                            28997,
                            1759,
                            1120,
                            29005,
                            139,
                            28999,
                            119,
                            102,
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
                            1752,
                            5650,
                            119,
                            29003,
                            13832,
                            3121,
                            2340,
                            144,
                            28997,
                            1759,
                            1120,
                            29005,
                            145,
                            28999,
                            119,
                            1262,
                            1771,
                            146,
                            119,
                            102,
                        ],
                    ]
                ),
            )
        else:
            torch.testing.assert_close(
                inputs.input_ids,
                torch.tensor(
                    [
                        [
                            101,
                            28998,
                            13832,
                            3121,
                            2340,
                            138,
                            28996,
                            1759,
                            1120,
                            28999,
                            139,
                            28997,
                            119,
                            102,
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
                            1752,
                            5650,
                            119,
                            28998,
                            13832,
                            3121,
                            2340,
                            144,
                            28996,
                            1759,
                            1120,
                            28999,
                            145,
                            28997,
                            119,
                            1262,
                            1771,
                            146,
                            119,
                            102,
                        ],
                    ]
                ),
            )
        torch.testing.assert_close(
            inputs.attention_mask,
            torch.tensor(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ]
            ),
        )

    if encode_target:
        torch.testing.assert_close(targets, torch.tensor([2, 2]))
    else:
        assert targets is None


def test_unbatch_output(prepared_taskmodule, model_output):
    unbatched_outputs = prepared_taskmodule.unbatch_output(model_output)

    assert len(unbatched_outputs) == 8

    labels = [
        "org:founded_by",
        "no_relation",
        "per:employee_of",
        "per:founder",
        "org:founded_by",
        "no_relation",
        "no_relation",
        "no_relation",
    ]
    probabilities = [0.6, 0.5, 0.6, 0.5, 0.4, 0.5, 0.6, 0.5]

    for output, label, probability in zip(unbatched_outputs, labels, probabilities):
        assert set(output.keys()) == {"labels", "probabilities"}
        assert output["labels"] == [label]
        assert output["probabilities"] == pytest.approx([probability])


@pytest.mark.parametrize("inplace", [False, True])
def test_decode(prepared_taskmodule, documents, model_output, inplace):
    documents = [documents[i] for i in [0, 1, 4]]

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

    expected_scores = [0.6, 0.5, 0.6, 0.5, 0.4, 0.5, 0.6, 0.5]
    i = 0
    for document in decoded_documents:
        for relation_expected, relation_decoded in zip(
            document["entities"], document["entities"].predictions
        ):
            assert relation_expected.start == relation_decoded.start
            assert relation_expected.end == relation_decoded.end
            assert relation_expected.label == relation_decoded.label
            assert expected_scores[i] == pytest.approx(relation_decoded.score)
            i += 1

    if not inplace:
        for document in documents:
            assert not document["relations"].predictions


def test_encode_with_partition(documents):
    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = TransformerRETextClassificationTaskModule(
        tokenizer_name_or_path=tokenizer_name_or_path,
        relation_annotation="relations",
        partition_annotation="sentences",
    )
    assert not taskmodule.is_from_pretrained
    taskmodule.prepare(documents)

    assert len(documents) == 8
    encodings = taskmodule.encode(documents)
    tokens = [
        taskmodule.tokenizer.convert_ids_to_tokens(encoding.inputs["input_ids"])
        for encoding in encodings
    ]
    assert len(encodings) == 5
    assert encodings[0].document != encodings[1].document
    assert encodings[1].document != encodings[2].document
    # the last document contains 3 valid relations
    assert encodings[2].document == encodings[3].document
    assert encodings[3].document == encodings[4].document
    assert tokens[0] == [
        "[CLS]",
        "[H]",
        "En",
        "##ti",
        "##ty",
        "A",
        "[/H]",
        "works",
        "at",
        "[T]",
        "B",
        "[/T]",
        ".",
        "[SEP]",
    ]
    assert tokens[1] == [
        "[CLS]",
        "[H]",
        "En",
        "##ti",
        "##ty",
        "G",
        "[/H]",
        "works",
        "at",
        "[T]",
        "H",
        "[/T]",
        ".",
        "[SEP]",
    ]
    assert tokens[2] == [
        "[CLS]",
        "[H]",
        "En",
        "##ti",
        "##ty",
        "M",
        "[/H]",
        "works",
        "at",
        "[T]",
        "N",
        "[/T]",
        ".",
        "[SEP]",
    ]
    assert tokens[3] == [
        "[CLS]",
        "And",
        "[H]",
        "it",
        "[/H]",
        "founded",
        "[T]",
        "O",
        "[/T]",
        "[SEP]",
    ]
    assert tokens[4] == [
        "[CLS]",
        "And",
        "[T]",
        "it",
        "[/T]",
        "founded",
        "[H]",
        "O",
        "[/H]",
        "[SEP]",
    ]


def test_encode_with_windowing(documents):
    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = TransformerRETextClassificationTaskModule(
        tokenizer_name_or_path=tokenizer_name_or_path,
        relation_annotation="relations",
        max_window=12,
    )
    assert not taskmodule.is_from_pretrained
    taskmodule.prepare(documents)

    assert len(documents) == 8
    encodings = taskmodule.encode(documents)
    assert len(encodings) == 3
    for encoding in encodings:
        assert len(encoding.inputs["input_ids"]) <= taskmodule.max_window
    tokens = [
        taskmodule.tokenizer.convert_ids_to_tokens(encoding.inputs["input_ids"])
        for encoding in encodings
    ]
    assert tokens[0] == [
        "[CLS]",
        "at",
        "[T]",
        "H",
        "[/T]",
        ".",
        "And",
        "founded",
        "[H]",
        "I",
        "[/H]",
        "[SEP]",
    ]
    assert tokens[1] == [
        "[CLS]",
        ".",
        "And",
        "[H]",
        "it",
        "[/H]",
        "founded",
        "[T]",
        "O",
        "[/T]",
        ".",
        "[SEP]",
    ]
    assert tokens[2] == [
        "[CLS]",
        ".",
        "And",
        "[T]",
        "it",
        "[/T]",
        "founded",
        "[H]",
        "O",
        "[/H]",
        ".",
        "[SEP]",
    ]
