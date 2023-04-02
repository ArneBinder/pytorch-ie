from copy import copy
from dataclasses import dataclass

import pytest
import torch
from transformers import BatchEncoding

from pytorch_ie import TransformerSeq2SeqTaskModule
from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.documents import TextDocument


@pytest.fixture(scope="module")
def taskmodule():
    transformer_model = "Babelscape/rebel-large"
    taskmodule = TransformerSeq2SeqTaskModule(tokenizer_name_or_path=transformer_model)
    assert not taskmodule.is_from_pretrained

    return taskmodule


def test_taskmodule(taskmodule):
    assert taskmodule is not None


@dataclass
class ExampleDocument(TextDocument):
    entities: AnnotationList[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")


@pytest.fixture(scope="module")
def document():
    result = ExampleDocument(
        "“Making a super tasty alt-chicken wing is only half of it,” said Po Bronson, general partner at SOSV "
        "and managing director of IndieBio."
    )
    # add a made-up relation
    head = LabeledSpan(start=22, end=38, label="food")
    tail = LabeledSpan(start=10, end=15, label="taste")
    rel = BinaryRelation(head=head, tail=tail, label="has_taste")
    result.entities.append(head)
    result.entities.append(tail)
    result.relations.append(rel)
    return result


@pytest.fixture(scope="module")
def task_encoding_without_targets(taskmodule, document):
    result = taskmodule.encode_input(document)
    return result


def test_encode_input(task_encoding_without_targets, document, taskmodule):
    assert task_encoding_without_targets is not None
    assert task_encoding_without_targets.document == document
    assert not task_encoding_without_targets.has_targets
    expected_input_tokens = [
        "<s>",
        "âĢ",
        "ľ",
        "Making",
        "Ġa",
        "Ġsuper",
        "Ġtasty",
        "Ġalt",
        "-",
        "ch",
        "icken",
        "Ġwing",
        "Ġis",
        "Ġonly",
        "Ġhalf",
        "Ġof",
        "Ġit",
        ",",
        "âĢ",
        "Ŀ",
        "Ġsaid",
        "ĠPo",
        "ĠBr",
        "onson",
        ",",
        "Ġgeneral",
        "Ġpartner",
        "Ġat",
        "ĠSOS",
        "V",
        "Ġand",
        "Ġmanaging",
        "Ġdirector",
        "Ġof",
        "ĠIndie",
        "Bio",
        ".",
        "</s>",
    ]
    assert set(task_encoding_without_targets.inputs) == {"input_ids", "attention_mask"}
    input_tokens = taskmodule.tokenizer.convert_ids_to_tokens(
        task_encoding_without_targets.inputs["input_ids"]
    )
    assert input_tokens == expected_input_tokens
    assert task_encoding_without_targets.inputs["attention_mask"] == [1] * len(
        expected_input_tokens
    )
    assert task_encoding_without_targets.metadata == {}


@pytest.fixture(scope="module")
def target(taskmodule, task_encoding_without_targets):
    return taskmodule.encode_target(task_encoding_without_targets)


def test_target(target, taskmodule):
    expected_label_tokens = [
        "<s>",
        "<triplet>",
        "Ġalt",
        "-",
        "ch",
        "icken",
        "Ġwing",
        "Ġ",
        "<subj>",
        "Ġsuper",
        "Ġ",
        "<obj>",
        "Ġhas",
        "_",
        "t",
        "aste",
        "</s>",
    ]
    assert set(target) == {"labels"}
    label_tokens = taskmodule.tokenizer.convert_ids_to_tokens(target["labels"])
    assert label_tokens == expected_label_tokens


@pytest.fixture(scope="module")
def task_encoding(task_encoding_without_targets, target):
    result = copy(task_encoding_without_targets)
    result.targets = target
    return result


@pytest.fixture(scope="module")
def model_predict_output():
    return torch.IntTensor(
        [
            [
                0,
                50267,
                36363,
                846,
                1437,
                50266,
                35890,
                40790,
                1437,
                50265,
                8540,
                1437,
                50267,
                35890,
                40790,
                1437,
                50266,
                36363,
                846,
                1437,
                50265,
                4095,
                1651,
                2,
            ]
        ],
    )


@pytest.fixture(scope="module")
def unbatched_outputs(taskmodule, model_predict_output):
    result = taskmodule.unbatch_output(model_predict_output)
    return result


def test_unpatch_output(unbatched_outputs):
    assert unbatched_outputs == [
        [
            {"head": "SOSV", "type": "subsidiary", "tail": "IndieBio"},
            {"head": "IndieBio", "type": "parent organization", "tail": "SOSV"},
        ]
    ]


@pytest.fixture(scope="module")
def annotations_from_output(taskmodule, task_encoding_without_targets, unbatched_outputs):
    task_encodings = [task_encoding_without_targets]
    assert len(task_encodings) == len(unbatched_outputs)
    named_annotations = []
    for task_encoding, unbatched_output in zip(task_encodings, unbatched_outputs):
        named_annotations.extend(
            taskmodule.create_annotations_from_output(
                task_encoding=task_encoding, task_output=unbatched_output
            )
        )
    return named_annotations


def test_annotations_from_output(annotations_from_output):
    assert annotations_from_output is not None
    assert len(annotations_from_output) == 6
    assert annotations_from_output[0] == (
        "entities",
        LabeledSpan(start=96, end=100, label="head", score=1.0),
    )
    assert annotations_from_output[1] == (
        "entities",
        LabeledSpan(start=126, end=134, label="tail", score=1.0),
    )
    assert annotations_from_output[2] == (
        "relations",
        BinaryRelation(
            head=LabeledSpan(start=96, end=100, label="head", score=1.0),
            tail=LabeledSpan(start=126, end=134, label="tail", score=1.0),
            label="subsidiary",
            score=1.0,
        ),
    )
    assert annotations_from_output[3] == (
        "entities",
        LabeledSpan(start=126, end=134, label="head", score=1.0),
    )
    assert annotations_from_output[4] == (
        "entities",
        LabeledSpan(start=96, end=100, label="tail", score=1.0),
    )
    assert annotations_from_output[5] == (
        "relations",
        BinaryRelation(
            head=LabeledSpan(start=126, end=134, label="head", score=1.0),
            tail=LabeledSpan(start=96, end=100, label="tail", score=1.0),
            label="parent organization",
            score=1.0,
        ),
    )


@pytest.fixture(scope="module")
def batch(taskmodule, task_encoding):
    return taskmodule.collate(task_encodings=[task_encoding])


def test_collate(batch):
    assert batch is not None
    # each batch consists of just one entry (this is not the number of batches!)
    assert len(batch) == 1
    batch_encoding = batch[0]

    input_ids_expected = torch.IntTensor(
        [
            [
                0,
                17,
                48,
                31845,
                10,
                2422,
                22307,
                11838,
                12,
                611,
                13552,
                5897,
                16,
                129,
                457,
                9,
                24,
                6,
                17,
                46,
                26,
                6002,
                2265,
                26942,
                6,
                937,
                1784,
                23,
                36363,
                846,
                8,
                4196,
                736,
                9,
                35890,
                40790,
                4,
                2,
            ],
        ]
    ).to(dtype=torch.int64)
    attention_mask_expected = torch.IntTensor(
        [
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ],
        ]
    ).to(dtype=torch.int64)
    labels_expected = torch.IntTensor(
        [
            [
                0,
                50267,
                11838,
                12,
                611,
                13552,
                5897,
                1437,
                50266,
                2422,
                1437,
                50265,
                34,
                1215,
                90,
                14631,
                2,
            ],
        ]
    ).to(dtype=torch.int64)

    encoding_expected = BatchEncoding(
        data={
            "input_ids": input_ids_expected,
            "attention_mask": attention_mask_expected,
            "labels": labels_expected,
        }
    )
    assert set(batch_encoding.data) == set(encoding_expected.data)
    torch.testing.assert_close(batch_encoding.input_ids, encoding_expected.input_ids)
    torch.testing.assert_close(batch_encoding.attention_mask, encoding_expected.attention_mask)
    torch.testing.assert_close(batch_encoding.labels, encoding_expected.labels)
