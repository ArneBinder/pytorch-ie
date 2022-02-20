import copy
import os

import numpy as np
import pytest
import torch
from numpy.testing import assert_almost_equal

from pytorch_ie.data import LabeledSpan
from pytorch_ie.taskmodules import TransformerTokenClassificationTaskModule
from pytorch_ie.taskmodules.transformer_token_classification import (
    convert_span_annotations_to_tag_sequence,
    _enumerate_windows,
)
from tests.taskmodules.document import (
    DOC1_ENTITY_BERLIN,
    DOC1_ENTITY_JANE,
    DOC1_ENTITY_KARL,
    DOC2_ENTITY_JENNY,
    DOC2_ENTITY_SEATTLE,
    DOC3_ENTITY_BERLIN,
    DOC3_ENTITY_KARL,
    get_doc1,
    get_doc2,
    get_doc3,
)


@pytest.fixture
def documents():
    doc_kwargs = dict(
        entity_annotation_name="entities",
        relation_annotation_name="relations",
        sentence_annotation_name="sentences",
    )
    documents = [get_doc1(**doc_kwargs), get_doc2(**doc_kwargs), get_doc3(**doc_kwargs)]
    return documents


@pytest.fixture(scope="module")
def taskmodule():
    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = TransformerTokenClassificationTaskModule(
        tokenizer_name_or_path=tokenizer_name_or_path,
    )
    return taskmodule


@pytest.fixture
def prepared_taskmodule(taskmodule, documents):
    taskmodule.prepare(documents)
    return taskmodule


@pytest.fixture(scope="module")
def taskmodule_with_partition():
    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = TransformerTokenClassificationTaskModule(
        tokenizer_name_or_path=tokenizer_name_or_path, partition_annotation="sentences"
    )
    return taskmodule


@pytest.fixture
def prepared_taskmodule_with_partition(taskmodule_with_partition, documents):
    taskmodule_with_partition.prepare(documents)
    return taskmodule_with_partition


@pytest.fixture(scope="module")
def taskmodule_with_window():
    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = TransformerTokenClassificationTaskModule(
        tokenizer_name_or_path=tokenizer_name_or_path,
        max_window=10,
    )
    return taskmodule


@pytest.fixture
def prepared_taskmodule_with_window(taskmodule_with_window, documents):
    taskmodule_with_window.prepare(documents)
    return taskmodule_with_window


@pytest.fixture(scope="module")
def taskmodule_with_window_overlap():
    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = TransformerTokenClassificationTaskModule(
        tokenizer_name_or_path=tokenizer_name_or_path,
        max_window=16,
        window_overlap=3,
    )
    return taskmodule


@pytest.fixture
def prepared_taskmodule_with_window_overlap(taskmodule_with_window_overlap, documents):
    taskmodule_with_window_overlap.prepare(documents)
    return taskmodule_with_window_overlap


def test_prepare(taskmodule, documents):
    # assert not taskmodule.is_prepared()
    taskmodule.prepare(documents)
    # assert taskmodule.is_prepared()
    assert taskmodule.label_to_id.keys() == {"O", "B-person", "I-person", "B-city", "I-city"}


def test_config(prepared_taskmodule):
    config = prepared_taskmodule._config()
    assert config["taskmodule_type"] == "TransformerTokenClassificationTaskModule"
    assert "label_to_id" in config
    assert set(config["label_to_id"]) == {"O", "B-person", "I-person", "B-city", "I-city"}


def test_encode_input(prepared_taskmodule, documents):
    (
        input_encoding,
        all_metadata,
        new_documents,
    ) = prepared_taskmodule.encode_input(documents)
    assert len(input_encoding) == 3
    assert len(all_metadata) == 3
    # assert new_documents is not None
    # assert len(new_documents) == 3

    encoding = input_encoding[0]
    document = documents[0]
    metadata = all_metadata[0]
    assert (
        prepared_taskmodule.tokenizer.convert_ids_to_tokens(encoding["input_ids"])
        == document.metadata["tokens"]
    )
    assert metadata["offset_mapping"] == [
        (0, 0),
        (0, 4),
        (5, 10),
        (11, 13),
        (14, 20),
        (20, 21),
        (22, 26),
        (27, 29),
        (30, 32),
        (33, 41),
        (42, 47),
        (48, 52),
        (0, 0),
    ]
    assert metadata["special_tokens_mask"] == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

    encoding = input_encoding[1]
    document = documents[1]
    metadata = all_metadata[1]
    assert (
        prepared_taskmodule.tokenizer.convert_ids_to_tokens(encoding["input_ids"])
        == document.metadata["tokens"]
    )
    assert metadata["offset_mapping"] == [
        (0, 0),
        (0, 7),
        (8, 10),
        (11, 12),
        (13, 18),
        (19, 23),
        (23, 24),
        (25, 30),
        (31, 33),
        (33, 36),
        (36, 37),
        (38, 40),
        (41, 44),
        (45, 49),
        (49, 50),
        (50, 51),
        (52, 57),
        (57, 58),
        (0, 0),
    ]
    assert metadata["special_tokens_mask"] == [
        1,
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
        1,
    ]

    encoding = input_encoding[2]
    document = documents[2]
    metadata = all_metadata[2]
    assert (
        prepared_taskmodule.tokenizer.convert_ids_to_tokens(encoding["input_ids"])
        == document.metadata["tokens"]
    )
    assert metadata["offset_mapping"] == [
        (0, 0),
        (0, 4),
        (5, 11),
        (12, 17),
        (18, 22),
        (23, 25),
        (26, 32),
        (32, 33),
        (0, 0),
    ]
    assert metadata["special_tokens_mask"] == [1, 0, 0, 0, 0, 0, 0, 0, 1]


@pytest.mark.parametrize("encode_target", [False, True])
def test_encode_with_partition(prepared_taskmodule, documents, encode_target):
    prepared_taskmodule_with_partition = copy.deepcopy(prepared_taskmodule)
    prepared_taskmodule_with_partition.single_sentence = True
    prepared_taskmodule_with_partition.sentence_annotation = "sentences"
    task_encodings = prepared_taskmodule_with_partition.encode(
        documents, encode_target=encode_target
    )
    assert len(task_encodings) == 4
    task_encoding = task_encodings[0]
    assert task_encoding.document == documents[0]
    assert prepared_taskmodule_with_partition.tokenizer.convert_ids_to_tokens(
        task_encoding.input["input_ids"]
    ) == ["[CLS]", "Jane", "lives", "in", "Berlin", ".", "[SEP]"]
    assert task_encoding.input["input_ids"] == [101, 4074, 2491, 1107, 3206, 119, 102]
    assert task_encoding.metadata["offset_mapping"] == [
        (0, 0),
        (0, 4),
        (5, 10),
        (11, 13),
        (14, 20),
        (20, 21),
        (0, 0),
    ]
    assert task_encoding.metadata["special_tokens_mask"] == [1, 0, 0, 0, 0, 0, 1]

    task_encoding = task_encodings[1]
    assert task_encoding.document == documents[1]
    assert prepared_taskmodule_with_partition.tokenizer.convert_ids_to_tokens(
        task_encoding.input["input_ids"]
    ) == ["[CLS]", "Seattle", "is", "a", "rainy", "city", ".", "[SEP]"]
    assert task_encoding.input["input_ids"] == [101, 5160, 1110, 170, 21098, 1331, 119, 102]
    assert task_encoding.metadata["offset_mapping"] == [
        (0, 0),
        (0, 7),
        (8, 10),
        (11, 12),
        (13, 18),
        (19, 23),
        (23, 24),
        (0, 0),
    ]
    assert task_encoding.metadata["special_tokens_mask"] == [1, 0, 0, 0, 0, 0, 0, 1]

    task_encoding = task_encodings[2]
    assert task_encoding.document == documents[1]
    assert prepared_taskmodule_with_partition.tokenizer.convert_ids_to_tokens(
        task_encoding.input["input_ids"]
    ) == [
        "[CLS]",
        "Jenny",
        "Du",
        "##rka",
        "##n",
        "is",
        "the",
        "city",
        "'",
        "s",
        "mayor",
        ".",
        "[SEP]",
    ]
    assert task_encoding.input["input_ids"] == [
        101,
        8067,
        12786,
        23870,
        1179,
        1110,
        1103,
        1331,
        112,
        188,
        4398,
        119,
        102,
    ]
    assert task_encoding.metadata["offset_mapping"] == [
        (0, 0),
        (0, 5),
        (6, 8),
        (8, 11),
        (11, 12),
        (13, 15),
        (16, 19),
        (20, 24),
        (24, 25),
        (25, 26),
        (27, 32),
        (32, 33),
        (0, 0),
    ]
    assert task_encoding.metadata["special_tokens_mask"] == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

    task_encoding = task_encodings[3]
    assert task_encoding.document == documents[2]
    assert prepared_taskmodule_with_partition.tokenizer.convert_ids_to_tokens(
        task_encoding.input["input_ids"]
    ) == ["[CLS]", "Karl", "enjoys", "sunny", "days", "in", "Berlin", ".", "[SEP]"]
    assert task_encoding.input["input_ids"] == [
        101,
        5728,
        16615,
        21162,
        1552,
        1107,
        3206,
        119,
        102,
    ]
    assert task_encoding.metadata["offset_mapping"] == [
        (0, 0),
        (0, 4),
        (5, 11),
        (12, 17),
        (18, 22),
        (23, 25),
        (26, 32),
        (32, 33),
        (0, 0),
    ]
    assert task_encoding.metadata["special_tokens_mask"] == [1, 0, 0, 0, 0, 0, 0, 0, 1]

    if encode_target:
        assert task_encodings[0].has_target
        assert task_encodings[0].target == [-100, 3, 0, 0, 1, 0, -100]

        assert task_encodings[1].has_target
        assert task_encodings[1].target == [-100, 1, 0, 0, 0, 0, 0, -100]

        assert task_encodings[2].has_target
        assert task_encodings[2].target == [-100, 3, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, -100]

        assert task_encodings[3].has_target
        assert task_encodings[3].target == [-100, 3, 0, 0, 0, 0, 1, 0, -100]


def test_enumerate_windows():
    sequence = [
        "Jane",
        "lives",
        "in",
        "Berlin",
        ".",
        "this",
        "is",
        "no",
        "sentence",
        "about",
        "Karl",
    ]
    windows = list(_enumerate_windows(sequence=sequence, max_size=5))
    assert len(windows) == 3
    token_slice, label_slice = windows[0]
    assert token_slice == (0, 5)
    assert label_slice == (0, 5)

    token_slice, label_slice = windows[1]
    assert token_slice == (5, 10)
    assert label_slice == (0, 5)

    token_slice, label_slice = windows[2]
    assert token_slice == (10, 11)
    assert label_slice == (0, 1)


def test_enumerate_windows_with_overlap():
    sequence = [
        "Jane",
        "lives",
        "in",
        "Berlin",
        ".",
        "this",
        "is",
        "no",
        "sentence",
        "about",
        "Karl",
    ]
    windows = list(_enumerate_windows(sequence=sequence, max_size=7, overlap=2))
    assert len(windows) == 3
    token_slice, label_slice = windows[0]
    assert token_slice == (0, 7)
    assert label_slice == (0, 5)

    token_slice, label_slice = windows[1]
    assert token_slice == (3, 10)
    assert label_slice == (2, 5)

    token_slice, label_slice = windows[2]
    assert token_slice == (6, 11)
    assert label_slice == (2, 5)


def test_enumerate_windows_with_overlap2():
    sequence = [
        "Seattle",
        "is",
        "a",
        "rainy",
        "city",
        ".",
        "Jenny",
        "Du",
        "##rka",
        "##n",
        "is",
        "the",
        "city",
        "'",
        "s",
        "mayor",
        ".",
    ]
    windows = list(_enumerate_windows(sequence=sequence, max_size=14, overlap=3))
    assert len(windows) == 2
    token_slice, label_slice = windows[0]
    assert token_slice == (0, 14)
    assert label_slice == (0, 11)

    token_slice, label_slice = windows[1]
    assert token_slice == (8, 17)
    assert label_slice == (3, 9)


def test_encode_input_with_window(prepared_taskmodule_with_window, documents):
    # prepared_taskmodule_with_window.max_window = 10
    (
        input_encodings,
        all_metadata,
        new_documents,
    ) = prepared_taskmodule_with_window.encode_input(documents)
    tokens = [
        prepared_taskmodule_with_window.tokenizer.convert_ids_to_tokens(
            input_encoding["input_ids"]
        )
        for input_encoding in input_encodings
    ]
    assert tokens[0] == [
        "[CLS]",
        "Jane",
        "lives",
        "in",
        "Berlin",
        ".",
        "this",
        "is",
        "no",
        "[SEP]",
    ]
    assert tokens[1] == ["[CLS]", "sentence", "about", "Karl", "[SEP]"]
    assert tokens[2] == [
        "[CLS]",
        "Seattle",
        "is",
        "a",
        "rainy",
        "city",
        ".",
        "Jenny",
        "Du",
        "[SEP]",
    ]
    assert tokens[3] == ["[CLS]", "##rka", "##n", "is", "the", "city", "'", "s", "mayor", "[SEP]"]
    assert tokens[4] == ["[CLS]", ".", "[SEP]"]
    assert tokens[5] == ["[CLS]", "Karl", "enjoys", "sunny", "days", "in", "Berlin", ".", "[SEP]"]
    # TODO: check metadata


def test_encode_input_with_window_and_overlap(prepared_taskmodule_with_window_overlap, documents):
    (
        input_encodings,
        all_metadata,
        new_documents,
    ) = prepared_taskmodule_with_window_overlap.encode_input(documents)
    tokens = [
        prepared_taskmodule_with_window_overlap.tokenizer.convert_ids_to_tokens(
            input_encoding["input_ids"]
        )
        for input_encoding in input_encodings
    ]
    assert tokens == [
        [
            "[CLS]",
            "Jane",
            "lives",
            "in",
            "Berlin",
            ".",
            "this",
            "is",
            "no",
            "sentence",
            "about",
            "Karl",
            "[SEP]",
        ],
        [
            "[CLS]",
            "Seattle",
            "is",
            "a",
            "rainy",
            "city",
            ".",
            "Jenny",
            "Du",
            "##rka",
            "##n",
            "is",
            "the",
            "city",
            "'",
            "[SEP]",
        ],
        ["[CLS]", "##rka", "##n", "is", "the", "city", "'", "s", "mayor", ".", "[SEP]"],
        ["[CLS]", "Karl", "enjoys", "sunny", "days", "in", "Berlin", ".", "[SEP]"],
    ]
    # TODO: check metadata


@pytest.mark.parametrize("encode_target", [False, True])
def test_encode_target_with_window(prepared_taskmodule_with_window, documents, encode_target):
    # prepared_taskmodule_with_window.max_window = 10
    task_encodings = prepared_taskmodule_with_window.encode(documents, encode_target=encode_target)
    tokens = [
        prepared_taskmodule_with_window.tokenizer.convert_ids_to_tokens(
            encoding.input["input_ids"]
        )
        for encoding in task_encodings
    ]
    assert len(task_encodings) == 6
    if encode_target:
        targets = [
            [
                prepared_taskmodule_with_window.id_to_label.get(target, None)
                for target in encoding.target
            ]
            for encoding in task_encodings
        ]
        assert tokens[0] == [
            "[CLS]",
            "Jane",
            "lives",
            "in",
            "Berlin",
            ".",
            "this",
            "is",
            "no",
            "[SEP]",
        ]
        assert targets[0] == [None, "B-person", "O", "O", "B-city", "O", "O", "O", "O", None]
        assert tokens[1] == ["[CLS]", "sentence", "about", "Karl", "[SEP]"]
        assert targets[1] == [None, "O", "O", "B-person", None]
        assert tokens[2] == [
            "[CLS]",
            "Seattle",
            "is",
            "a",
            "rainy",
            "city",
            ".",
            "Jenny",
            "Du",
            "[SEP]",
        ]
        assert targets[2] == [None, "B-city", "O", "O", "O", "O", "O", "O", "O", None]
        assert tokens[3] == [
            "[CLS]",
            "##rka",
            "##n",
            "is",
            "the",
            "city",
            "'",
            "s",
            "mayor",
            "[SEP]",
        ]
        assert targets[3] == [None, "O", "O", "O", "O", "O", "O", "O", "O", None]
        assert tokens[4] == ["[CLS]", ".", "[SEP]"]
        assert targets[4] == [None, "O", None]
        assert tokens[5] == [
            "[CLS]",
            "Karl",
            "enjoys",
            "sunny",
            "days",
            "in",
            "Berlin",
            ".",
            "[SEP]",
        ]
        assert targets[5] == [None, "B-person", "O", "O", "O", "O", "B-city", "O", None]
    else:
        assert [encoding.has_target for encoding in task_encodings] == [False] * len(
            task_encodings
        )


@pytest.mark.parametrize("encode_target", [False, True])
def test_encode_target_with_window_and_overlap(
    prepared_taskmodule_with_window_overlap, documents, encode_target
):
    task_encodings = prepared_taskmodule_with_window_overlap.encode(
        documents, encode_target=encode_target
    )
    tokens = [
        prepared_taskmodule_with_window_overlap.tokenizer.convert_ids_to_tokens(
            encoding.input["input_ids"]
        )
        for encoding in task_encodings
    ]
    assert len(task_encodings) == 4
    if encode_target:
        targets = [
            [
                prepared_taskmodule_with_window_overlap.id_to_label.get(target, None)
                for target in encoding.target
            ]
            for encoding in task_encodings
        ]
        assert tokens[0] == [
            "[CLS]",
            "Jane",
            "lives",
            "in",
            "Berlin",
            ".",
            "this",
            "is",
            "no",
            "sentence",
            "about",
            "Karl",
            "[SEP]",
        ]
        assert targets[0] == [
            None,
            "B-person",
            "O",
            "O",
            "B-city",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "B-person",
            None,
        ]

        assert tokens[1] == [
            "[CLS]",
            "Seattle",
            "is",
            "a",
            "rainy",
            "city",
            ".",
            "Jenny",
            "Du",
            "##rka",
            "##n",
            "is",
            "the",
            "city",
            "'",
            "[SEP]",
        ]
        assert targets[1] == [
            None,
            "B-city",
            "O",
            "O",
            "O",
            "O",
            "O",
            "B-person",
            "I-person",
            "I-person",
            "I-person",
            "O",
            None,
            None,
            None,
            None,
        ]

        assert tokens[2] == [
            "[CLS]",
            "##rka",
            "##n",
            "is",
            "the",
            "city",
            "'",
            "s",
            "mayor",
            ".",
            "[SEP]",
        ]
        assert targets[2] == [None, None, None, None, "O", "O", "O", "O", "O", "O", None]

        assert tokens[3] == [
            "[CLS]",
            "Karl",
            "enjoys",
            "sunny",
            "days",
            "in",
            "Berlin",
            ".",
            "[SEP]",
        ]
        assert targets[3] == [None, "B-person", "O", "O", "O", "O", "B-city", "O", None]

    else:
        assert [encoding.has_target for encoding in task_encodings] == [False] * len(
            task_encodings
        )


@pytest.mark.parametrize("encode_target", [False, True])
def test_encode_target(prepared_taskmodule, documents, encode_target):
    task_encodings = prepared_taskmodule.encode(documents, encode_target=encode_target)
    assert len(task_encodings) == 3
    if encode_target:
        encoding = task_encodings[0]
        assert encoding.has_target
        target_labels = [prepared_taskmodule.id_to_label.get(_id, _id) for _id in encoding.target]
        # labels for special tokens are not used for training and set to -100
        assert target_labels == [
            -100,
            "B-person",
            "O",
            "O",
            "B-city",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "B-person",
            -100,
        ]

        encoding = task_encodings[1]
        assert encoding.has_target
        target_labels = [prepared_taskmodule.id_to_label.get(_id, _id) for _id in encoding.target]
        # labels for special tokens are not used for training and set to -100
        assert target_labels == [
            -100,
            "B-city",
            "O",
            "O",
            "O",
            "O",
            "O",
            "B-person",
            "I-person",
            "I-person",
            "I-person",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            -100,
        ]

        encoding = task_encodings[2]
        assert encoding.has_target
        target_labels = [prepared_taskmodule.id_to_label.get(_id, _id) for _id in encoding.target]
        # labels for special tokens are not used for training and set to -100
        assert target_labels == [-100, "B-person", "O", "O", "O", "O", "B-city", "O", -100]
    else:
        assert [encoding.has_target for encoding in task_encodings] == [False, False, False]


@pytest.mark.parametrize("encode_target", [False, True])
def test_encode(prepared_taskmodule, documents, encode_target):
    # the code is actually tested in test_encode_input() and test_encode_target(). Here we only test assertions in encode().
    task_encodings = prepared_taskmodule.encode(documents, encode_target=True)


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
    input_ids = inputs["input_ids"].tolist()
    attention_mask = inputs["attention_mask"].tolist()
    assert prepared_taskmodule.tokenizer.convert_ids_to_tokens(input_ids[0]) == [
        "[CLS]",
        "Jane",
        "lives",
        "in",
        "Berlin",
        ".",
        "this",
        "is",
        "no",
        "sentence",
        "about",
        "Karl",
        "[SEP]",
        "[PAD]",
        "[PAD]",
        "[PAD]",
        "[PAD]",
        "[PAD]",
        "[PAD]",
    ]
    assert prepared_taskmodule.tokenizer.convert_ids_to_tokens(input_ids[1]) == [
        "[CLS]",
        "Seattle",
        "is",
        "a",
        "rainy",
        "city",
        ".",
        "Jenny",
        "Du",
        "##rka",
        "##n",
        "is",
        "the",
        "city",
        "'",
        "s",
        "mayor",
        ".",
        "[SEP]",
    ]
    assert prepared_taskmodule.tokenizer.convert_ids_to_tokens(input_ids[2]) == [
        "[CLS]",
        "Karl",
        "enjoys",
        "sunny",
        "days",
        "in",
        "Berlin",
        ".",
        "[SEP]",
        "[PAD]",
        "[PAD]",
        "[PAD]",
        "[PAD]",
        "[PAD]",
        "[PAD]",
        "[PAD]",
        "[PAD]",
        "[PAD]",
        "[PAD]",
    ]
    assert input_ids[0] == [
        101,
        4074,
        2491,
        1107,
        3206,
        119,
        1142,
        1110,
        1185,
        5650,
        1164,
        5728,
        102,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    assert input_ids[1] == [
        101,
        5160,
        1110,
        170,
        21098,
        1331,
        119,
        8067,
        12786,
        23870,
        1179,
        1110,
        1103,
        1331,
        112,
        188,
        4398,
        119,
        102,
    ]
    assert input_ids[2] == [
        101,
        5728,
        16615,
        21162,
        1552,
        1107,
        3206,
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
    ]
    assert attention_mask[0] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    assert attention_mask[1] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert attention_mask[2] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    if encode_target:
        assert targets.shape == (3, 19)
        labels = [
            [prepared_taskmodule.id_to_label.get(target_id, target_id) for target_id in target_ids]
            for target_ids in targets.tolist()
        ]
        assert labels[0] == [
            -100,
            "B-person",
            "O",
            "O",
            "B-city",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "B-person",
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
        ]
        assert labels[1] == [
            -100,
            "B-city",
            "O",
            "O",
            "O",
            "O",
            "O",
            "B-person",
            "I-person",
            "I-person",
            "I-person",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            -100,
        ]
        assert labels[2] == [
            -100,
            "B-person",
            "O",
            "O",
            "O",
            "O",
            "B-city",
            "O",
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
        ]
    else:
        assert targets is None


@pytest.fixture
def model_output():
    return {
        "logits": torch.tensor(
            [
                [
                    [9.2733, -1.0300, -2.5785, -1.0300, -2.5785],
                    [-1.6924, 9.5473, -1.9625, -9.5473, -1.9625],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                ],
                [
                    [9.2733, -1.0300, -2.5785, -1.0300, -2.5785],
                    [-1.6924, 9.5473, -1.9625, -9.5473, -1.9625],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                ],
                [
                    [9.2733, -1.0300, -2.5785, -1.0300, -2.5785],
                    [-1.6924, 9.5473, -1.9625, -9.5473, -1.9625],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                    [-0.9995, -2.5705, 10.0095, -0.9995, -2.5705],
                ],
            ]
        ),
    }


def test_unbatch_output(prepared_taskmodule, model_output):
    unbatched_outputs = prepared_taskmodule.unbatch_output(model_output)
    assert len(unbatched_outputs) == 3

    unbatched_output = unbatched_outputs[0]
    assert unbatched_output["tags"] == [
        "O",
        "B-city",
        "I-city",
        "I-city",
        "I-city",
        "I-city",
        "I-city",
        "I-city",
        "I-city",
        "I-city",
        "I-city",
        "I-city",
        "I-city",
        "I-city",
        "I-city",
        "I-city",
        "I-city",
        "I-city",
        "I-city",
        "I-city",
        "I-city",
    ]
    assert_almost_equal(
        unbatched_output["probabilities"],
        np.array(
            [
                [
                    0.9999186992645264,
                    3.351956547703594e-05,
                    7.125139291019877e-06,
                    3.351956547703594e-05,
                    7.125139291019877e-06,
                ],
                [
                    1.3141525414539501e-05,
                    0.9999668598175049,
                    1.0030970770458225e-05,
                    5.09689757066667e-09,
                    1.0030970770458225e-05,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
            ]
        ),
    )

    unbatched_output = unbatched_outputs[1]
    assert unbatched_output["tags"] == [
        "O",
        "B-city",
        "I-city",
        "I-city",
        "I-city",
        "I-city",
        "I-city",
        "I-city",
        "I-city",
        "I-city",
        "I-city",
        "I-city",
        "I-city",
        "I-city",
        "I-city",
        "I-city",
        "I-city",
        "I-city",
        "I-city",
        "I-city",
        "I-city",
    ]
    assert_almost_equal(
        unbatched_output["probabilities"],
        np.array(
            [
                [
                    0.9999186992645264,
                    3.351956547703594e-05,
                    7.125139291019877e-06,
                    3.351956547703594e-05,
                    7.125139291019877e-06,
                ],
                [
                    1.3141525414539501e-05,
                    0.9999668598175049,
                    1.0030970770458225e-05,
                    5.09689757066667e-09,
                    1.0030970770458225e-05,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
            ]
        ),
    )

    unbatched_output = unbatched_outputs[2]
    assert unbatched_output["tags"] == [
        "O",
        "B-city",
        "I-city",
        "I-city",
        "I-city",
        "I-city",
        "I-city",
        "I-city",
        "I-city",
        "I-city",
        "I-city",
        "I-city",
        "I-city",
        "I-city",
        "I-city",
        "I-city",
        "I-city",
        "I-city",
        "I-city",
        "I-city",
        "I-city",
    ]
    assert_almost_equal(
        unbatched_output["probabilities"],
        np.array(
            [
                [
                    0.9999186992645264,
                    3.351956547703594e-05,
                    7.125139291019877e-06,
                    3.351956547703594e-05,
                    7.125139291019877e-06,
                ],
                [
                    1.3141525414539501e-05,
                    0.9999668598175049,
                    1.0030970770458225e-05,
                    5.09689757066667e-09,
                    1.0030970770458225e-05,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
                [
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                    0.9999599456787109,
                    1.655140113143716e-05,
                    3.439997044552001e-06,
                ],
            ]
        ),
    )


@pytest.mark.parametrize("inplace", [True, False])
def test_decode(prepared_taskmodule, documents, model_output, inplace):
    encodings = prepared_taskmodule.encode(documents, encode_target=False)
    unbatched_outputs = prepared_taskmodule.unbatch_output(model_output)
    decoded_documents = prepared_taskmodule.decode(
        encodings=encodings,
        decoded_outputs=unbatched_outputs,
        input_documents=documents,
        inplace=inplace,
    )

    assert len(decoded_documents) == len(documents)
    if inplace:
        assert set(decoded_documents) == set(documents)
    else:
        assert set(decoded_documents).isdisjoint(set(documents))

    decoded_document = decoded_documents[0]
    predictions = decoded_document.predictions("entities")
    assert len(predictions) == 1
    entity = predictions[0]
    assert entity.label == "city"
    assert entity.start == 0
    assert entity.end == 52

    decoded_document = decoded_documents[1]
    predictions = decoded_document.predictions("entities")
    assert len(predictions) == 1
    entity = predictions[0]
    assert entity.label == "city"
    assert entity.start == 0
    assert entity.end == 58

    decoded_document = decoded_documents[2]
    predictions = decoded_document.predictions("entities")
    assert len(predictions) == 1
    entity = predictions[0]
    assert entity.label == "city"
    assert entity.start == 0
    assert entity.end == 33


def test_save_load(tmp_path, prepared_taskmodule):
    path = os.path.join(tmp_path, "taskmodule")
    prepared_taskmodule.save_pretrained(path)
    loaded_taskmodule = TransformerTokenClassificationTaskModule.from_pretrained(path)
    # assert loaded_taskmodule.is_prepared()
    assert loaded_taskmodule._config() == prepared_taskmodule._config()


def test_convert_span_annotations_to_tag_sequence(taskmodule, documents):
    doc = documents[0]
    entities = doc.span_annotations("entities")
    assert len(entities) == 3
    encoding = taskmodule.encode_text(text=doc.text)
    tag_sequence = convert_span_annotations_to_tag_sequence(
        spans=entities,
        special_tokens_mask=encoding["special_tokens_mask"],
        char_to_token_mapper=encoding.char_to_token,
    )
    assert tag_sequence == [
        None,
        "B-person",
        "O",
        "O",
        "B-city",
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
        "B-person",
        None,
    ]

    doc = documents[1]
    entities = doc.span_annotations("entities")
    assert len(entities) == 2
    encoding = taskmodule.encode_text(text=doc.text)
    tag_sequence = convert_span_annotations_to_tag_sequence(
        spans=entities,
        special_tokens_mask=encoding["special_tokens_mask"],
        char_to_token_mapper=encoding.char_to_token,
    )
    assert tag_sequence == [
        None,
        "B-city",
        "O",
        "O",
        "O",
        "O",
        "O",
        "B-person",
        "I-person",
        "I-person",
        "I-person",
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
        None,
    ]

    doc = documents[2]
    entities = doc.span_annotations("entities")
    assert len(entities) == 2
    encoding = taskmodule.encode_text(text=doc.text)
    tag_sequence = convert_span_annotations_to_tag_sequence(
        spans=entities,
        special_tokens_mask=encoding["special_tokens_mask"],
        char_to_token_mapper=encoding.char_to_token,
    )
    assert tag_sequence == [None, "B-person", "O", "O", "O", "O", "B-city", "O", None]


def test_convert_span_annotations_to_tag_sequence_with_partition(
    taskmodule_with_partition, documents
):
    doc = documents[0]
    entities = doc.span_annotations("entities")
    assert len(entities) == 3
    partitions = doc.span_annotations("sentences")
    assert len(partitions) == 1
    partition = partitions[0]
    encoding = taskmodule_with_partition.encode_text(text=doc.text, partition=partition)
    tag_sequence = convert_span_annotations_to_tag_sequence(
        spans=entities,
        special_tokens_mask=encoding["special_tokens_mask"],
        char_to_token_mapper=encoding.char_to_token,
        partition=partition,
    )
    assert tag_sequence == [None, "B-person", "O", "O", "B-city", "O", None]

    doc = documents[1]
    entities = doc.span_annotations("entities")
    assert len(entities) == 2
    partitions = doc.span_annotations("sentences")
    assert len(partitions) == 2
    partition = partitions[0]
    encoding = taskmodule_with_partition.encode_text(text=doc.text, partition=partition)
    tag_sequence = convert_span_annotations_to_tag_sequence(
        spans=entities,
        special_tokens_mask=encoding["special_tokens_mask"],
        char_to_token_mapper=encoding.char_to_token,
        partition=partition,
    )
    assert tag_sequence == [None, "B-city", "O", "O", "O", "O", "O", None]
    partition = partitions[1]
    encoding = taskmodule_with_partition.encode_text(text=doc.text, partition=partition)
    tag_sequence = convert_span_annotations_to_tag_sequence(
        spans=entities,
        special_tokens_mask=encoding["special_tokens_mask"],
        char_to_token_mapper=encoding.char_to_token,
        partition=partition,
    )
    assert tag_sequence == [
        None,
        "B-person",
        "I-person",
        "I-person",
        "I-person",
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
        None,
    ]

    doc = documents[2]
    entities = doc.span_annotations("entities")
    assert len(entities) == 2
    partitions = doc.span_annotations("sentences")
    assert len(partitions) == 1
    partition = partitions[0]
    encoding = taskmodule_with_partition.encode_text(text=doc.text, partition=partition)
    tag_sequence = convert_span_annotations_to_tag_sequence(
        spans=entities,
        special_tokens_mask=encoding["special_tokens_mask"],
        char_to_token_mapper=encoding.char_to_token,
        partition=partition,
    )
    assert tag_sequence == [None, "B-person", "O", "O", "O", "O", "B-city", "O", None]


def assert_entity_matches(actual: LabeledSpan, expected: LabeledSpan):
    assert actual.label == expected.label
    assert actual.start == expected.start
    assert actual.end == expected.end


def test_create_annotations_from_output(prepared_taskmodule, documents):
    assert prepared_taskmodule.entity_annotation == "entities"
    input_encodings = prepared_taskmodule.encode(documents, encode_target=False)
    input_encoding = input_encodings[0]
    """
    # create expected outputs via (NOTE: this requires encode_target=True)
    outputs = [
        {
            "tags": [
                prepared_taskmodule.id_to_label.get(_id, "SPECIAL_TOKEN")
                for _id in input_encoding.target
            ]
        }
        for input_encoding in input_encodings
    ]
    """
    outputs = [
        {
            "tags": [
                "SPECIAL_TOKEN",
                "B-person",
                "O",
                "O",
                "B-city",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "B-person",
                "SPECIAL_TOKEN",
            ]
        },
        {
            "tags": [
                "SPECIAL_TOKEN",
                "B-city",
                "O",
                "O",
                "O",
                "O",
                "O",
                "B-person",
                "I-person",
                "I-person",
                "I-person",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "SPECIAL_TOKEN",
            ]
        },
        {
            "tags": [
                "SPECIAL_TOKEN",
                "B-person",
                "O",
                "O",
                "O",
                "O",
                "B-city",
                "O",
                "SPECIAL_TOKEN",
            ]
        },
    ]
    output = outputs[0]
    predicted_entities = list(
        prepared_taskmodule.create_annotations_from_output(encoding=input_encoding, output=output)
    )
    assert len(predicted_entities) == 3
    predicted_entities_sorted = sorted(
        predicted_entities, key=lambda name_and_annotation: name_and_annotation[1].start
    )
    predicted_name_and_span = predicted_entities_sorted[0]
    assert predicted_name_and_span[0] == prepared_taskmodule.entity_annotation
    assert_entity_matches(predicted_name_and_span[1], DOC1_ENTITY_JANE)
    predicted_name_and_span = predicted_entities_sorted[1]
    assert predicted_name_and_span[0] == prepared_taskmodule.entity_annotation
    assert_entity_matches(predicted_name_and_span[1], DOC1_ENTITY_BERLIN)
    predicted_name_and_span = predicted_entities_sorted[2]
    assert predicted_name_and_span[0] == prepared_taskmodule.entity_annotation
    assert_entity_matches(predicted_name_and_span[1], DOC1_ENTITY_KARL)

    input_encoding = input_encodings[1]
    output = outputs[1]
    predicted_entities = list(
        prepared_taskmodule.create_annotations_from_output(encoding=input_encoding, output=output)
    )
    assert len(predicted_entities) == 2
    predicted_entities_sorted = sorted(
        predicted_entities, key=lambda name_and_annotation: name_and_annotation[1].start
    )
    predicted_name_and_span = predicted_entities_sorted[0]
    assert predicted_name_and_span[0] == prepared_taskmodule.entity_annotation
    assert_entity_matches(predicted_name_and_span[1], DOC2_ENTITY_SEATTLE)
    predicted_name_and_span = predicted_entities_sorted[1]
    assert predicted_name_and_span[0] == prepared_taskmodule.entity_annotation
    assert_entity_matches(predicted_name_and_span[1], DOC2_ENTITY_JENNY)

    input_encoding = input_encodings[2]
    output = outputs[2]
    predicted_entities = list(
        prepared_taskmodule.create_annotations_from_output(encoding=input_encoding, output=output)
    )
    assert len(predicted_entities) == 2
    predicted_entities_sorted = sorted(
        predicted_entities, key=lambda name_and_annotation: name_and_annotation[1].start
    )
    predicted_name_and_span = predicted_entities_sorted[0]
    assert predicted_name_and_span[0] == prepared_taskmodule.entity_annotation
    assert_entity_matches(predicted_name_and_span[1], DOC3_ENTITY_KARL)
    predicted_name_and_span = predicted_entities_sorted[1]
    assert predicted_name_and_span[0] == prepared_taskmodule.entity_annotation
    assert_entity_matches(predicted_name_and_span[1], DOC3_ENTITY_BERLIN)


def test_create_annotations_from_output_with_partition(
    prepared_taskmodule_with_partition, documents
):
    assert prepared_taskmodule_with_partition.entity_annotation == "entities"
    input_encodings = prepared_taskmodule_with_partition.encode(documents, encode_target=False)
    outputs = [
        {"tags": ["SPECIAL_TOKEN", "B-person", "O", "O", "B-city", "O", "SPECIAL_TOKEN"]},
        {"tags": ["SPECIAL_TOKEN", "B-city", "O", "O", "O", "O", "O", "SPECIAL_TOKEN"]},
        {
            "tags": [
                "SPECIAL_TOKEN",
                "B-person",
                "I-person",
                "I-person",
                "I-person",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "SPECIAL_TOKEN",
            ]
        },
        {
            "tags": [
                "SPECIAL_TOKEN",
                "B-person",
                "O",
                "O",
                "O",
                "O",
                "B-city",
                "O",
                "SPECIAL_TOKEN",
            ]
        },
    ]
    output = outputs[0]
    input_encoding = input_encodings[0]
    predicted_entities = list(
        prepared_taskmodule_with_partition.create_annotations_from_output(
            encoding=input_encoding, output=output
        )
    )
    assert len(predicted_entities) == 2
    predicted_entities_sorted = sorted(
        predicted_entities, key=lambda name_and_annotation: name_and_annotation[1].start
    )
    predicted_name_and_span = predicted_entities_sorted[0]
    assert (
        predicted_name_and_span[0] == prepared_taskmodule_with_partition.entity_annotation
    )  # prepared_taskmodule_with_partition.entity_annotation
    assert_entity_matches(predicted_name_and_span[1], DOC1_ENTITY_JANE)
    predicted_name_and_span = predicted_entities_sorted[1]
    assert (
        predicted_name_and_span[0] == prepared_taskmodule_with_partition.entity_annotation
    )  # prepared_taskmodule_with_partition.entity_annotation
    assert_entity_matches(predicted_name_and_span[1], DOC1_ENTITY_BERLIN)

    input_encoding = input_encodings[1]
    output = outputs[1]
    predicted_entities = list(
        prepared_taskmodule_with_partition.create_annotations_from_output(
            encoding=input_encoding, output=output
        )
    )
    assert len(predicted_entities) == 1
    predicted_entities_sorted = sorted(
        predicted_entities, key=lambda name_and_annotation: name_and_annotation[1].start
    )
    predicted_name_and_span = predicted_entities_sorted[0]
    assert (
        predicted_name_and_span[0] == prepared_taskmodule_with_partition.entity_annotation
    )  # prepared_taskmodule_with_partition.entity_annotation
    assert_entity_matches(predicted_name_and_span[1], DOC2_ENTITY_SEATTLE)

    input_encoding = input_encodings[2]
    output = outputs[2]
    predicted_entities = list(
        prepared_taskmodule_with_partition.create_annotations_from_output(
            encoding=input_encoding, output=output
        )
    )
    assert len(predicted_entities) == 1
    predicted_entities_sorted = sorted(
        predicted_entities, key=lambda name_and_annotation: name_and_annotation[1].start
    )
    predicted_name_and_span = predicted_entities_sorted[0]
    assert predicted_name_and_span[0] == prepared_taskmodule_with_partition.entity_annotation
    assert_entity_matches(predicted_name_and_span[1], DOC2_ENTITY_JENNY)

    input_encoding = input_encodings[3]
    output = outputs[3]
    predicted_entities = list(
        prepared_taskmodule_with_partition.create_annotations_from_output(
            encoding=input_encoding, output=output
        )
    )
    assert len(predicted_entities) == 2
    predicted_entities_sorted = sorted(
        predicted_entities, key=lambda name_and_annotation: name_and_annotation[1].start
    )
    predicted_name_and_span = predicted_entities_sorted[0]
    assert (
        predicted_name_and_span[0] == prepared_taskmodule_with_partition.entity_annotation
    )  # prepared_taskmodule_with_partition.entity_annotation
    assert_entity_matches(predicted_name_and_span[1], DOC3_ENTITY_KARL)
    predicted_name_and_span = predicted_entities_sorted[1]
    assert (
        predicted_name_and_span[0] == prepared_taskmodule_with_partition.entity_annotation
    )  # prepared_taskmodule_with_partition.entity_annotation
    assert_entity_matches(predicted_name_and_span[1], DOC3_ENTITY_BERLIN)


def test_create_annotations_from_output_with_window(prepared_taskmodule_with_window, documents):
    assert prepared_taskmodule_with_window.entity_annotation == "entities"
    input_encodings = prepared_taskmodule_with_window.encode(documents, encode_target=False)
    """
    # create expected outputs via (NOTE: this requires encode_target=True)
    outputs = [
        {
            "tags": [
                prepared_taskmodule_with_window.id_to_label.get(_id, "SPECIAL_TOKEN")
                for _id in input_encoding.target
            ]
        }
        for input_encoding in input_encodings
    ]
    """
    outputs = [
        {
            "tags": [
                "SPECIAL_TOKEN",
                "B-person",
                "O",
                "O",
                "B-city",
                "O",
                "O",
                "O",
                "O",
                "SPECIAL_TOKEN",
            ]
        },
        {"tags": ["SPECIAL_TOKEN", "O", "O", "B-person", "SPECIAL_TOKEN"]},
        {"tags": ["SPECIAL_TOKEN", "B-city", "O", "O", "O", "O", "O", "O", "O", "SPECIAL_TOKEN"]},
        {"tags": ["SPECIAL_TOKEN", "O", "O", "O", "O", "O", "O", "O", "O", "SPECIAL_TOKEN"]},
        {"tags": ["SPECIAL_TOKEN", "O", "SPECIAL_TOKEN"]},
        {
            "tags": [
                "SPECIAL_TOKEN",
                "B-person",
                "O",
                "O",
                "O",
                "O",
                "B-city",
                "O",
                "SPECIAL_TOKEN",
            ]
        },
    ]

    output = outputs[0]
    input_encoding = input_encodings[0]
    predicted_entities = list(
        prepared_taskmodule_with_window.create_annotations_from_output(
            encoding=input_encoding, output=output
        )
    )
    assert len(predicted_entities) == 2
    predicted_entities_sorted = sorted(
        predicted_entities, key=lambda name_and_annotation: name_and_annotation[1].start
    )
    predicted_name_and_span = predicted_entities_sorted[0]
    assert (
        predicted_name_and_span[0] == prepared_taskmodule_with_window.entity_annotation
    )  # prepared_taskmodule_with_partition.entity_annotation
    assert_entity_matches(predicted_name_and_span[1], DOC1_ENTITY_JANE)
    predicted_name_and_span = predicted_entities_sorted[1]
    assert (
        predicted_name_and_span[0] == prepared_taskmodule_with_window.entity_annotation
    )  # prepared_taskmodule_with_partition.entity_annotation
    assert_entity_matches(predicted_name_and_span[1], DOC1_ENTITY_BERLIN)

    output = outputs[1]
    input_encoding = input_encodings[1]
    predicted_entities = list(
        prepared_taskmodule_with_window.create_annotations_from_output(
            encoding=input_encoding, output=output
        )
    )
    assert len(predicted_entities) == 1
    predicted_entities_sorted = sorted(
        predicted_entities, key=lambda name_and_annotation: name_and_annotation[1].start
    )
    predicted_name_and_span = predicted_entities_sorted[0]
    assert (
        predicted_name_and_span[0] == prepared_taskmodule_with_window.entity_annotation
    )  # prepared_taskmodule_with_partition.entity_annotation
    assert_entity_matches(predicted_name_and_span[1], DOC1_ENTITY_KARL)

    output = outputs[2]
    input_encoding = input_encodings[2]
    predicted_entities = list(
        prepared_taskmodule_with_window.create_annotations_from_output(
            encoding=input_encoding, output=output
        )
    )
    assert len(predicted_entities) == 1
    predicted_entities_sorted = sorted(
        predicted_entities, key=lambda name_and_annotation: name_and_annotation[1].start
    )
    predicted_name_and_span = predicted_entities_sorted[0]
    assert (
        predicted_name_and_span[0] == prepared_taskmodule_with_window.entity_annotation
    )  # prepared_taskmodule_with_partition.entity_annotation
    assert_entity_matches(predicted_name_and_span[1], DOC2_ENTITY_SEATTLE)

    output = outputs[3]
    input_encoding = input_encodings[3]
    predicted_entities = list(
        prepared_taskmodule_with_window.create_annotations_from_output(
            encoding=input_encoding, output=output
        )
    )
    assert len(predicted_entities) == 0

    output = outputs[4]
    input_encoding = input_encodings[4]
    predicted_entities = list(
        prepared_taskmodule_with_window.create_annotations_from_output(
            encoding=input_encoding, output=output
        )
    )
    assert len(predicted_entities) == 0

    output = outputs[5]
    input_encoding = input_encodings[5]
    predicted_entities = list(
        prepared_taskmodule_with_window.create_annotations_from_output(
            encoding=input_encoding, output=output
        )
    )
    assert len(predicted_entities) == 2
    predicted_entities_sorted = sorted(
        predicted_entities, key=lambda name_and_annotation: name_and_annotation[1].start
    )
    predicted_name_and_span = predicted_entities_sorted[0]
    assert (
        predicted_name_and_span[0] == prepared_taskmodule_with_window.entity_annotation
    )  # prepared_taskmodule_with_partition.entity_annotation
    assert_entity_matches(predicted_name_and_span[1], DOC3_ENTITY_KARL)
    predicted_name_and_span = predicted_entities_sorted[1]
    assert (
        predicted_name_and_span[0] == prepared_taskmodule_with_window.entity_annotation
    )  # prepared_taskmodule_with_partition.entity_annotation
    assert_entity_matches(predicted_name_and_span[1], DOC3_ENTITY_BERLIN)


def test_create_annotations_from_output_with_window_overlap(
    prepared_taskmodule_with_window_overlap, documents
):
    assert prepared_taskmodule_with_window_overlap.entity_annotation == "entities"
    input_encodings = prepared_taskmodule_with_window_overlap.encode(
        documents, encode_target=False
    )
    """
    # create expected outputs via (NOTE: this requires encode_target=True)
    outputs = [
        {
            "tags": [
                prepared_taskmodule_with_window_overlap.id_to_label.get(_id, "SPECIAL_TOKEN")
                for _id in input_encoding.target
            ]
        }
        for input_encoding in input_encodings
    ]
    """
    outputs = [
        {
            "tags": [
                "SPECIAL_TOKEN",
                "B-person",
                "O",
                "O",
                "B-city",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "B-person",
                "SPECIAL_TOKEN",
            ]
        },
        {
            "tags": [
                "SPECIAL_TOKEN",
                "B-city",
                "O",
                "O",
                "O",
                "O",
                "O",
                "B-person",
                "I-person",
                "I-person",
                "I-person",
                "O",
                "O",
                "O",
                "O",
                "SPECIAL_TOKEN",
            ]
        },
        {"tags": ["SPECIAL_TOKEN", "O", "O", "O", "O", "O", "O", "O", "O", "O", "SPECIAL_TOKEN"]},
        {
            "tags": [
                "SPECIAL_TOKEN",
                "B-person",
                "O",
                "O",
                "O",
                "O",
                "B-city",
                "O",
                "SPECIAL_TOKEN",
            ]
        },
    ]

    output = outputs[0]
    input_encoding = input_encodings[0]
    predicted_entities = list(
        prepared_taskmodule_with_window_overlap.create_annotations_from_output(
            encoding=input_encoding, output=output
        )
    )
    assert len(predicted_entities) == 3
    predicted_entities_sorted = sorted(
        predicted_entities, key=lambda name_and_annotation: name_and_annotation[1].start
    )
    predicted_name_and_span = predicted_entities_sorted[0]
    assert (
        predicted_name_and_span[0] == prepared_taskmodule_with_window_overlap.entity_annotation
    )  # prepared_taskmodule_with_partition.entity_annotation
    assert_entity_matches(predicted_name_and_span[1], DOC1_ENTITY_JANE)
    predicted_name_and_span = predicted_entities_sorted[1]
    assert (
        predicted_name_and_span[0] == prepared_taskmodule_with_window_overlap.entity_annotation
    )  # prepared_taskmodule_with_partition.entity_annotation
    assert_entity_matches(predicted_name_and_span[1], DOC1_ENTITY_BERLIN)
    predicted_name_and_span = predicted_entities_sorted[2]
    assert (
        predicted_name_and_span[0] == prepared_taskmodule_with_window_overlap.entity_annotation
    )  # prepared_taskmodule_with_partition.entity_annotation
    assert_entity_matches(predicted_name_and_span[1], DOC1_ENTITY_KARL)

    output = outputs[1]
    input_encoding = input_encodings[1]
    predicted_entities = list(
        prepared_taskmodule_with_window_overlap.create_annotations_from_output(
            encoding=input_encoding, output=output
        )
    )
    assert len(predicted_entities) == 2
    predicted_entities_sorted = sorted(
        predicted_entities, key=lambda name_and_annotation: name_and_annotation[1].start
    )
    predicted_name_and_span = predicted_entities_sorted[0]
    assert (
        predicted_name_and_span[0] == prepared_taskmodule_with_window_overlap.entity_annotation
    )  # prepared_taskmodule_with_partition.entity_annotation
    assert_entity_matches(predicted_name_and_span[1], DOC2_ENTITY_SEATTLE)
    predicted_name_and_span = predicted_entities_sorted[1]
    assert (
        predicted_name_and_span[0] == prepared_taskmodule_with_window_overlap.entity_annotation
    )  # prepared_taskmodule_with_partition.entity_annotation
    assert_entity_matches(predicted_name_and_span[1], DOC2_ENTITY_JENNY)

    output = outputs[2]
    input_encoding = input_encodings[2]
    predicted_entities = list(
        prepared_taskmodule_with_window_overlap.create_annotations_from_output(
            encoding=input_encoding, output=output
        )
    )
    assert len(predicted_entities) == 0

    output = outputs[3]
    input_encoding = input_encodings[3]
    predicted_entities = list(
        prepared_taskmodule_with_window_overlap.create_annotations_from_output(
            encoding=input_encoding, output=output
        )
    )
    assert len(predicted_entities) == 2
    predicted_entities_sorted = sorted(
        predicted_entities, key=lambda name_and_annotation: name_and_annotation[1].start
    )
    predicted_name_and_span = predicted_entities_sorted[0]
    assert (
        predicted_name_and_span[0] == prepared_taskmodule_with_window_overlap.entity_annotation
    )  # prepared_taskmodule_with_partition.entity_annotation
    assert_entity_matches(predicted_name_and_span[1], DOC3_ENTITY_KARL)
    predicted_name_and_span = predicted_entities_sorted[1]
    assert (
        predicted_name_and_span[0] == prepared_taskmodule_with_window_overlap.entity_annotation
    )  # prepared_taskmodule_with_partition.entity_annotation
    assert_entity_matches(predicted_name_and_span[1], DOC3_ENTITY_BERLIN)
