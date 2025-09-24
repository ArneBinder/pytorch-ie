import logging
import pickle
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List

import pytest
import torch
from pie_core import AnnotationLayer, annotation_field
from pie_documents.annotations import LabeledSpan
from pie_documents.documents import (
    TextBasedDocument,
    TextDocumentWithLabeledSpans,
    TextDocumentWithLabeledSpansAndLabeledPartitions,
)
from torch import tensor
from transformers import BatchEncoding

from pytorch_ie.taskmodules import LabeledSpanExtractionByTokenClassificationTaskModule
from pytorch_ie.taskmodules.labeled_span_extraction_by_token_classification import ModelOutputType
from tests import _config_to_str

CONFIG_DEFAULT: dict[str, Any] = {}
CONFIG_MAX_WINDOW = {
    "tokenize_kwargs": {"max_length": 8, "truncation": True, "return_overflowing_tokens": True}
}
CONFIG_MAX_WINDOW_WITH_STRIDE = {
    "tokenize_kwargs": {
        "max_length": 8,
        "stride": 2,
        "truncation": True,
        "return_overflowing_tokens": True,
    }
}
CONFIG_PARTITIONS = {"partition_annotation": "sentences"}

CONFIGS: List[Dict[str, Any]] = [
    CONFIG_DEFAULT,
    CONFIG_MAX_WINDOW,
    CONFIG_MAX_WINDOW_WITH_STRIDE,
    CONFIG_PARTITIONS,
]

CONFIGS_DICT = {_config_to_str(cfg): cfg for cfg in CONFIGS}


@pytest.fixture(scope="module", params=CONFIGS_DICT.keys())
def config(request):
    """
    - Provides clean and readable test configurations.
    - Yields config dictionaries from the CONFIGS list to produce clean test case identifiers.

    """
    return CONFIGS_DICT[request.param]


@pytest.fixture(scope="module")
def config_str(config):
    # Fixture returning a string representation of the config
    return _config_to_str(config)


@pytest.fixture(scope="module")
def unprepared_taskmodule(config):
    """
    - Prepares a task module with the specified tokenizer and configuration.
    - Sets up the task module with a unprepared state for testing purposes.

    """
    return LabeledSpanExtractionByTokenClassificationTaskModule(
        tokenizer_name_or_path="bert-base-uncased", span_annotation="entities", **config
    )


@dataclass
class ExampleDocument(TextBasedDocument):
    entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
    sentences: AnnotationLayer[LabeledSpan] = annotation_field(target="text")


@pytest.fixture(scope="module")
def documents():
    """
    - Creates example documents with predefined texts.
    - Assigns labels to the documents for testing purposes.

    """
    doc1 = ExampleDocument(text="Mount Everest is the highest peak in the world.", id="doc1")
    doc1.entities.append(LabeledSpan(start=0, end=13, label="LOC"))
    assert str(doc1.entities[0]) == "Mount Everest"

    doc2 = ExampleDocument(text="Alice loves reading books. Bob enjoys playing soccer.", id="doc2")
    doc2.entities.append(LabeledSpan(start=0, end=5, label="PER"))
    assert str(doc2.entities[0]) == "Alice"
    doc2.entities.append(LabeledSpan(start=27, end=30, label="PER"))
    assert str(doc2.entities[1]) == "Bob"
    # we add just one sentence to doc2 that covers only Bob
    doc2.sentences.append(LabeledSpan(start=27, end=53, label="sentence"))
    assert str(doc2.sentences[0]) == "Bob enjoys playing soccer."

    return [doc1, doc2]


def test_taskmodule(unprepared_taskmodule):
    assert unprepared_taskmodule is not None


@pytest.fixture(scope="module")
def taskmodule(unprepared_taskmodule, documents):
    """
    - Prepares the task module with the given documents, i.e. collect available label values.
    - Calls the necessary methods to prepare the task module with the documents.
    - Calls _prepare(documents) and then _post_prepare()

    """
    unprepared_taskmodule.prepare(documents)
    return unprepared_taskmodule


def test_prepare(taskmodule):
    assert taskmodule is not None
    assert taskmodule.is_prepared
    assert taskmodule.label_to_id == {"B-LOC": 1, "B-PER": 3, "I-LOC": 2, "I-PER": 4, "O": 0}
    assert taskmodule.id_to_label == {0: "O", 1: "B-LOC", 2: "I-LOC", 3: "B-PER", 4: "I-PER"}


def test_config(taskmodule):
    config = taskmodule._config()
    assert config["taskmodule_type"] == "LabeledSpanExtractionByTokenClassificationTaskModule"
    assert "labels" in config
    assert config["labels"] == ["LOC", "PER"]


@pytest.fixture(scope="module")
def task_encodings_without_targets(taskmodule, documents):
    """
    - Generates task encodings for all the documents, but without associated targets.
    """
    return taskmodule.encode(documents, encode_target=False)


def test_task_encodings_without_targets(task_encodings_without_targets, taskmodule, config):
    tokens = [
        taskmodule.tokenizer.convert_ids_to_tokens(task_encoding.inputs.ids)
        for task_encoding in task_encodings_without_targets
    ]

    # If config is empty
    if config == CONFIG_DEFAULT:
        assert tokens == [
            [
                "[CLS]",
                "mount",
                "everest",
                "is",
                "the",
                "highest",
                "peak",
                "in",
                "the",
                "world",
                ".",
                "[SEP]",
            ],
            [
                "[CLS]",
                "alice",
                "loves",
                "reading",
                "books",
                ".",
                "bob",
                "enjoys",
                "playing",
                "soccer",
                ".",
                "[SEP]",
            ],
        ]

    # If config has the specified values (max_window=8, window_overlap=2)
    elif config == CONFIG_MAX_WINDOW_WITH_STRIDE:
        for t in tokens:
            assert len(t) <= 8

        assert tokens == [
            ["[CLS]", "mount", "everest", "is", "the", "highest", "peak", "[SEP]"],
            ["[CLS]", "highest", "peak", "in", "the", "world", ".", "[SEP]"],
            ["[CLS]", "alice", "loves", "reading", "books", ".", "bob", "[SEP]"],
            ["[CLS]", ".", "bob", "enjoys", "playing", "soccer", ".", "[SEP]"],
        ]

    # If config has the specified value (max_window=8)
    elif config == CONFIG_MAX_WINDOW:
        for t in tokens:
            assert len(t) <= 8

        assert tokens == [
            ["[CLS]", "mount", "everest", "is", "the", "highest", "peak", "[SEP]"],
            ["[CLS]", "in", "the", "world", ".", "[SEP]"],
            ["[CLS]", "alice", "loves", "reading", "books", ".", "bob", "[SEP]"],
            ["[CLS]", "enjoys", "playing", "soccer", ".", "[SEP]"],
        ]

    # If config has the specified value (partition_annotation=sentences)
    elif config == CONFIG_PARTITIONS:
        assert tokens

    else:
        raise ValueError(f"unknown config: {config}")


@pytest.fixture(scope="module")
def task_encodings(taskmodule, documents):
    return taskmodule.encode(documents, encode_target=True)


def test_task_encodings(task_encodings, taskmodule, config):
    tokens = [
        taskmodule.tokenizer.convert_ids_to_tokens(task_encoding.inputs.ids)
        for task_encoding in task_encodings
    ]
    labels_tokens = [
        [taskmodule.id_to_label[x] if x != -100 else "<pad>" for x in task_encoding.targets]
        for task_encoding in task_encodings
    ]
    assert len(labels_tokens) == len(tokens)

    tokens_with_labels = list(zip(tokens, labels_tokens))

    for tokens, labels in tokens_with_labels:
        assert len(tokens) == len(labels)

    # If config is empty
    if config == CONFIG_DEFAULT:
        assert tokens_with_labels == [
            (
                [
                    "[CLS]",
                    "mount",
                    "everest",
                    "is",
                    "the",
                    "highest",
                    "peak",
                    "in",
                    "the",
                    "world",
                    ".",
                    "[SEP]",
                ],
                ["<pad>", "B-LOC", "I-LOC", "O", "O", "O", "O", "O", "O", "O", "O", "<pad>"],
            ),
            (
                [
                    "[CLS]",
                    "alice",
                    "loves",
                    "reading",
                    "books",
                    ".",
                    "bob",
                    "enjoys",
                    "playing",
                    "soccer",
                    ".",
                    "[SEP]",
                ],
                ["<pad>", "B-PER", "O", "O", "O", "O", "B-PER", "O", "O", "O", "O", "<pad>"],
            ),
        ]

    # If config has the specified values (max_window=8, window_overlap=2)
    elif config == CONFIG_MAX_WINDOW_WITH_STRIDE:
        for tokens, labels in tokens_with_labels:
            assert len(tokens) <= 8

        assert tokens_with_labels == [
            (
                ["[CLS]", "mount", "everest", "is", "the", "highest", "peak", "[SEP]"],
                ["<pad>", "B-LOC", "I-LOC", "O", "O", "O", "O", "<pad>"],
            ),
            (
                ["[CLS]", "highest", "peak", "in", "the", "world", ".", "[SEP]"],
                ["<pad>", "O", "O", "O", "O", "O", "O", "<pad>"],
            ),
            (
                ["[CLS]", "alice", "loves", "reading", "books", ".", "bob", "[SEP]"],
                ["<pad>", "B-PER", "O", "O", "O", "O", "B-PER", "<pad>"],
            ),
            (
                ["[CLS]", ".", "bob", "enjoys", "playing", "soccer", ".", "[SEP]"],
                ["<pad>", "O", "B-PER", "O", "O", "O", "O", "<pad>"],
            ),
        ]

    # If config has the specified value (max_window=8)
    elif config == CONFIG_MAX_WINDOW:
        for tokens, labels in tokens_with_labels:
            assert len(tokens) <= 8

        assert tokens_with_labels == [
            (
                ["[CLS]", "mount", "everest", "is", "the", "highest", "peak", "[SEP]"],
                ["<pad>", "B-LOC", "I-LOC", "O", "O", "O", "O", "<pad>"],
            ),
            (
                ["[CLS]", "in", "the", "world", ".", "[SEP]"],
                ["<pad>", "O", "O", "O", "O", "<pad>"],
            ),
            (
                ["[CLS]", "alice", "loves", "reading", "books", ".", "bob", "[SEP]"],
                ["<pad>", "B-PER", "O", "O", "O", "O", "B-PER", "<pad>"],
            ),
            (
                ["[CLS]", "enjoys", "playing", "soccer", ".", "[SEP]"],
                ["<pad>", "O", "O", "O", "O", "<pad>"],
            ),
        ]

    # If config has the specified value (partition_annotation=sentences)
    elif config == CONFIG_PARTITIONS:
        assert tokens_with_labels == [
            (
                ["[CLS]", "bob", "enjoys", "playing", "soccer", ".", "[SEP]"],
                ["<pad>", "B-PER", "O", "O", "O", "O", "<pad>"],
            )
        ]

    else:
        raise ValueError(f"unknown config: {config}")


def test_encode_targets_with_overlap(caplog):
    # setup taskmodule
    taskmodule = LabeledSpanExtractionByTokenClassificationTaskModule(
        tokenizer_name_or_path="bert-base-uncased", labels=["LOC", "PER"]
    )
    taskmodule.post_prepare()

    # create a document with overlapping entities
    doc = TextDocumentWithLabeledSpans(
        text="Alice loves reading books. Bob enjoys playing soccer."
    )
    doc.labeled_spans.append(LabeledSpan(start=0, end=5, label="PER"))
    doc.labeled_spans.append(LabeledSpan(start=27, end=30, label="PER"))
    doc.labeled_spans.append(LabeledSpan(start=27, end=37, label="PER"))
    assert str(doc.labeled_spans[0]) == "Alice"
    assert str(doc.labeled_spans[1]) == "Bob"
    assert str(doc.labeled_spans[2]) == "Bob enjoys"

    # encode the document
    with caplog.at_level(logging.WARNING):
        task_encodings = taskmodule.encode([doc], encode_target=True)
    assert len(caplog.records) == 1
    assert (
        caplog.messages[0]
        == "tag already assigned (current span has an overlap: ('bob', 'enjoys'))."
    )
    assert len(task_encodings) == 1
    assert task_encodings[0].targets == [-100, 3, 0, 0, 0, 0, 3, 0, 0, 0, 0, -100]


@pytest.fixture(scope="module")
def task_encodings_for_batch(task_encodings, config):
    # just take everything we have
    return task_encodings


@pytest.fixture(scope="module")
def batch(taskmodule, task_encodings_for_batch, config) -> BatchEncoding:
    return taskmodule.collate(task_encodings_for_batch)


def test_collate(batch, config):
    assert batch is not None
    assert len(batch) == 2
    inputs, targets = batch

    assert set(inputs.data) == {"input_ids", "attention_mask", "special_tokens_mask"}
    input_ids_list = inputs.input_ids.tolist()
    attention_mask_list = inputs.attention_mask.tolist()
    special_tokens_mask_list = inputs.special_tokens_mask.tolist()
    assert set(targets) == {"labels"}
    labels_list = targets["labels"].tolist()

    # If config is empty
    if config == CONFIG_DEFAULT:
        assert input_ids_list == [
            [101, 4057, 23914, 2003, 1996, 3284, 4672, 1999, 1996, 2088, 1012, 102],
            [101, 5650, 7459, 3752, 2808, 1012, 3960, 15646, 2652, 4715, 1012, 102],
        ]
        assert attention_mask_list == [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
        assert labels_list == [
            [-100, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, -100],
            [-100, 3, 0, 0, 0, 0, 3, 0, 0, 0, 0, -100],
        ]
        assert special_tokens_mask_list == [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]

    # If config has the specified values (max_window=8, window_overlap=2)
    elif config == CONFIG_MAX_WINDOW_WITH_STRIDE:
        assert input_ids_list == [
            [101, 4057, 23914, 2003, 1996, 3284, 4672, 102],
            [101, 3284, 4672, 1999, 1996, 2088, 1012, 102],
            [101, 5650, 7459, 3752, 2808, 1012, 3960, 102],
            [101, 1012, 3960, 15646, 2652, 4715, 1012, 102],
        ]
        assert attention_mask_list == [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ]
        assert labels_list == [
            [-100, 1, 2, 0, 0, 0, 0, -100],
            [-100, 0, 0, 0, 0, 0, 0, -100],
            [-100, 3, 0, 0, 0, 0, 3, -100],
            [-100, 0, 3, 0, 0, 0, 0, -100],
        ]

    # If config has the specified values (max_window=8)
    elif config == CONFIG_MAX_WINDOW:
        assert input_ids_list == [
            [101, 4057, 23914, 2003, 1996, 3284, 4672, 102],
            [101, 1999, 1996, 2088, 1012, 102, 0, 0],
            [101, 5650, 7459, 3752, 2808, 1012, 3960, 102],
            [101, 15646, 2652, 4715, 1012, 102, 0, 0],
        ]
        assert attention_mask_list == [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 0],
        ]
        assert labels_list == [
            [-100, 1, 2, 0, 0, 0, 0, -100],
            [-100, 0, 0, 0, 0, -100, -100, -100],
            [-100, 3, 0, 0, 0, 0, 3, -100],
            [-100, 0, 0, 0, 0, -100, -100, -100],
        ]
        assert special_tokens_mask_list == [
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1, 1, 1],
        ]

    # If config has the specified value (partition_annotation=sentences)
    elif config == CONFIG_PARTITIONS:
        assert input_ids_list == [[101, 3960, 15646, 2652, 4715, 1012, 102]]
        assert attention_mask_list == [[1, 1, 1, 1, 1, 1, 1]]
        assert labels_list == [[-100, 3, 0, 0, 0, 0, -100]]
        assert special_tokens_mask_list == [[1, 0, 0, 0, 0, 0, 1]]

    else:
        raise ValueError(f"unknown config: {config}")

    inputs_expected = BatchEncoding(
        data={
            "input_ids": torch.tensor(input_ids_list, dtype=torch.int64),
            "attention_mask": torch.tensor(attention_mask_list, dtype=torch.int64),
            "special_tokens_mask": torch.tensor(special_tokens_mask_list, dtype=torch.int64),
        }
    )
    assert set(inputs.data) == set(inputs_expected.data)
    labels_expected = torch.tensor(labels_list, dtype=torch.int64)
    assert torch.equal(targets["labels"], labels_expected)


# This is not used, but can be used to create a batch of task encodings with targets for the unbatched_outputs fixture.
@pytest.fixture(scope="module")
def real_model_output(batch, taskmodule):
    from pytorch_ie.models import TransformerTokenClassificationModel

    model = TransformerTokenClassificationModel(
        model_name_or_path="prajjwal1/bert-tiny",
        num_classes=len(taskmodule.label_to_id),
    )
    inputs, targets = batch
    result = model(inputs)
    return result


@pytest.fixture(scope="module")
def model_output(config, batch, taskmodule) -> ModelOutputType:
    # create "perfect" output from targets
    labels = batch[1]["labels"]
    num_classes = len(taskmodule.label_to_id)
    # create one-hot encoding from labels
    labels_valid = labels.clone()
    labels_valid[labels_valid == taskmodule.label_pad_id] = taskmodule.label_to_id["O"]
    # create one-hot encoding from labels, but with 0.9 for the correct labels
    probabilities = (
        torch.nn.functional.one_hot(labels_valid, num_classes=num_classes).to(torch.float32) * 0.9
    )
    assert isinstance(labels, torch.LongTensor)
    assert isinstance(probabilities, torch.FloatTensor)
    return {"labels": labels, "probabilities": probabilities}


@pytest.fixture(scope="module")
def unbatched_outputs(taskmodule, model_output):
    return taskmodule.unbatch_output(model_output)


@pytest.mark.parametrize("combine_token_scores_method", ["mean", "max", "product", "UNKNOWN"])
def test_combine_token_scores_method(documents, combine_token_scores_method):
    taskmodule = LabeledSpanExtractionByTokenClassificationTaskModule(
        tokenizer_name_or_path="bert-base-uncased",
        span_annotation="entities",
        combine_token_scores_method=combine_token_scores_method,
    )
    taskmodule.prepare(documents)

    task_encodings = taskmodule.encode(documents, encode_target=True)
    batch = taskmodule.collate(task_encodings)

    # create "perfect" output from targets
    labels = batch[1]["labels"]
    num_classes = len(taskmodule.label_to_id)
    # create one-hot encoding from labels
    labels_valid = labels.clone()
    labels_valid[labels_valid == taskmodule.label_pad_id] = taskmodule.label_to_id["O"]
    # create one-hot encoding from labels, but with 0.9 for the correct labels
    probabilities = (
        torch.nn.functional.one_hot(labels_valid, num_classes=num_classes).to(torch.float32) * 0.9
    )
    # stepwise decrease the "winning" probabilities per token to test the different combine_token_scores_methods
    diff = 0.0
    for i in range(probabilities.size(1)):
        probabilities[:, i] -= diff
        diff += 0.01
    probabilities[probabilities < 0] = 0.0

    model_output = {"labels": labels, "probabilities": probabilities}

    unbatched_outputs = taskmodule.unbatch_output(model_output)

    if combine_token_scores_method == "UNKNOWN":
        with pytest.raises(ValueError) as excinfo:
            taskmodule.decode_annotations(unbatched_outputs[0])
        assert str(excinfo.value) == "combine_token_scores_method=UNKNOWN is not supported."
    else:
        annotations = []
        scores = []
        for unbatched_output in unbatched_outputs:
            decoded_annotations = taskmodule.decode_annotations(unbatched_output)
            assert set(decoded_annotations.keys()) == {"labeled_spans"}
            # Sort the annotations in each document by start and end position and label
            sorted_annotations = sorted(decoded_annotations["labeled_spans"])
            annotations.append(sorted_annotations)
            scores.append([round(ann.score, 5) for ann in sorted_annotations])

        # input values are (before combination): [[0.89, 0.88], [[0.89], [0.84]]]
        if combine_token_scores_method == "mean":
            assert scores == [[(0.89 + 0.88) / 2], [0.89, 0.84]]
        elif combine_token_scores_method == "max":
            assert scores == [[0.89], [0.89, 0.84]]
        elif combine_token_scores_method == "min":
            assert scores == [[0.88], [0.89, 0.84]]
        elif combine_token_scores_method == "product":
            assert scores == [[0.89 * 0.88], [0.89, 0.84]]
        else:
            raise ValueError(f"unknown combine_token_scores_method: {combine_token_scores_method}")


def test_unbatched_output(unbatched_outputs, config):
    assert unbatched_outputs is not None

    if config == CONFIG_DEFAULT:
        assert len(unbatched_outputs) == 2
        torch.testing.assert_close(
            unbatched_outputs[0]["labels"],
            torch.tensor([-100, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, -100]),
        )
        torch.testing.assert_close(
            unbatched_outputs[1]["labels"],
            torch.tensor([-100, 3, 0, 0, 0, 0, 3, 0, 0, 0, 0, -100]),
        )
    elif config == CONFIG_MAX_WINDOW_WITH_STRIDE:
        assert len(unbatched_outputs) == 4
        torch.testing.assert_close(
            unbatched_outputs[0]["labels"], torch.tensor([-100, 1, 2, 0, 0, 0, 0, -100])
        )
        torch.testing.assert_close(
            unbatched_outputs[1]["labels"], torch.tensor([-100, 0, 0, 0, 0, 0, 0, -100])
        )
        torch.testing.assert_close(
            unbatched_outputs[2]["labels"], torch.tensor([-100, 3, 0, 0, 0, 0, 3, -100])
        )
        torch.testing.assert_close(
            unbatched_outputs[3]["labels"], torch.tensor([-100, 0, 3, 0, 0, 0, 0, -100])
        )
    elif config == CONFIG_MAX_WINDOW:
        assert len(unbatched_outputs) == 4
        torch.testing.assert_close(
            unbatched_outputs[0]["labels"], torch.tensor([-100, 1, 2, 0, 0, 0, 0, -100])
        )
        torch.testing.assert_close(
            unbatched_outputs[1]["labels"], torch.tensor([-100, 0, 0, 0, 0, -100, -100, -100])
        )
        torch.testing.assert_close(
            unbatched_outputs[2]["labels"], torch.tensor([-100, 3, 0, 0, 0, 0, 3, -100])
        )
        torch.testing.assert_close(
            unbatched_outputs[3]["labels"], torch.tensor([-100, 0, 0, 0, 0, -100, -100, -100])
        )
    elif config == CONFIG_PARTITIONS:
        assert len(unbatched_outputs) == 1
        torch.testing.assert_close(
            unbatched_outputs[0]["labels"], torch.tensor([-100, 3, 0, 0, 0, 0, -100])
        )
    else:
        raise ValueError(f"unknown config: {config}")


def test_decode_annotations(taskmodule, unbatched_outputs, config):
    annotations = []
    for unbatched_output in unbatched_outputs:
        decoded_annotations = taskmodule.decode_annotations(unbatched_output)
        assert set(decoded_annotations.keys()) == {"labeled_spans"}
        # Sort the annotations in each document by start and end position and label
        annotations.append(
            sorted(
                decoded_annotations["labeled_spans"],
                key=lambda labeled_span: (
                    labeled_span.start,
                    labeled_span.end,
                    labeled_span.label,
                ),
            )
        )

    # Check based on the config
    if config == CONFIG_DEFAULT:
        assert annotations == [
            [LabeledSpan(start=1, end=3, label="LOC")],
            [
                LabeledSpan(start=1, end=2, label="PER"),
                LabeledSpan(start=6, end=7, label="PER"),
            ],
        ]

    elif config == CONFIG_MAX_WINDOW_WITH_STRIDE:
        # We get two annotations for Bob because the window overlaps with the previous one.
        # This is not a problem because annotations get de-duplicated during serialization.
        assert annotations == [
            [LabeledSpan(start=1, end=3, label="LOC")],
            [],
            [
                LabeledSpan(start=1, end=2, label="PER"),
                LabeledSpan(start=6, end=7, label="PER"),
            ],
            [LabeledSpan(start=2, end=3, label="PER")],
        ]

    elif config == CONFIG_MAX_WINDOW:
        assert annotations == [
            [LabeledSpan(start=1, end=3, label="LOC")],
            [],
            [
                LabeledSpan(start=1, end=2, label="PER"),
                LabeledSpan(start=6, end=7, label="PER"),
            ],
            [],
        ]

    elif config == CONFIG_PARTITIONS:
        assert annotations == [[LabeledSpan(start=1, end=2, label="PER", score=1.0)]]

    else:
        raise ValueError(f"unknown config: {config}")

    # assert that all scores are 0.9
    for doc_annotations in annotations:
        for annotation in doc_annotations:
            assert round(annotation.score, 4) == 0.9


@pytest.fixture(scope="module")
def annotations_from_output(taskmodule, task_encodings_for_batch, unbatched_outputs, config):
    named_annotations_per_document = defaultdict(list)
    for task_encoding, task_output in zip(task_encodings_for_batch, unbatched_outputs):
        annotations = taskmodule.create_annotations_from_output(task_encoding, task_output)
        named_annotations_per_document[task_encoding.document.id].extend(list(annotations))
    return named_annotations_per_document


def test_annotations_from_output(annotations_from_output, config, documents):
    assert annotations_from_output is not None
    # Sort the annotations in each document by start and end positions
    annotations_from_output = {
        doc_id: sorted(annotations, key=lambda x: (x[0], x[1].start, x[1].end))
        for doc_id, annotations in annotations_from_output.items()
    }
    documents_by_id = {doc.id: doc for doc in documents}
    documents_with_annotations = []
    resolved_annotations = defaultdict(list)
    # Check that the number of annotations is correct
    for doc_id, layer_names_and_annotations in annotations_from_output.items():
        new_doc = documents_by_id[doc_id].copy()
        for layer_name, annotation in layer_names_and_annotations:
            assert layer_name == "entities"
            assert isinstance(annotation, LabeledSpan)
            new_doc.entities.predictions.append(annotation)
            resolved_annotations[doc_id].append(str(annotation))
        documents_with_annotations.append(new_doc)

    resolved_annotations = dict(resolved_annotations)
    # Check based on the config
    if config == CONFIG_DEFAULT:
        assert resolved_annotations == {"doc1": ["Mount Everest"], "doc2": ["Alice", "Bob"]}

    elif config == CONFIG_MAX_WINDOW_WITH_STRIDE:
        # We get two annotations for Bob because the window overlaps with the previous one.
        # This is not a problem because annotations get de-duplicated during serialization.
        assert resolved_annotations == {"doc1": ["Mount Everest"], "doc2": ["Alice", "Bob", "Bob"]}

    elif config == CONFIG_MAX_WINDOW:
        assert resolved_annotations == {"doc1": ["Mount Everest"], "doc2": ["Alice", "Bob"]}

    elif config == CONFIG_PARTITIONS:
        assert resolved_annotations == {"doc2": ["Bob"]}

    else:
        raise ValueError(f"unknown config: {config}")


def test_document_type():
    taskmodule = LabeledSpanExtractionByTokenClassificationTaskModule(
        tokenizer_name_or_path="bert-base-uncased"
    )
    assert taskmodule.document_type == TextDocumentWithLabeledSpans


def test_document_type_with_partitions():
    taskmodule = LabeledSpanExtractionByTokenClassificationTaskModule(
        tokenizer_name_or_path="bert-base-uncased", partition_annotation="labeled_partitions"
    )
    assert taskmodule.document_type == TextDocumentWithLabeledSpansAndLabeledPartitions


def test_document_type_with_non_default_span_annotation(caplog):
    with caplog.at_level(logging.WARNING):
        taskmodule = LabeledSpanExtractionByTokenClassificationTaskModule(
            tokenizer_name_or_path="bert-base-uncased", span_annotation="entities"
        )
    assert taskmodule.document_type is None
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert (
        caplog.records[0].message
        == "span_annotation=entities is not the default value ('labeled_spans'), so the taskmodule "
        "LabeledSpanExtractionByTokenClassificationTaskModule can not request the usual document type "
        "(TextDocumentWithLabeledSpans) for auto-conversion because this has the bespoken default value "
        "as layer name(s) instead of the provided one(s)."
    )


def test_document_type_with_non_default_partition_annotation(caplog):
    with caplog.at_level(logging.WARNING):
        taskmodule = LabeledSpanExtractionByTokenClassificationTaskModule(
            tokenizer_name_or_path="bert-base-uncased", partition_annotation="sentences"
        )
    assert taskmodule.document_type is None
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert (
        caplog.records[0].message
        == "partition_annotation=sentences is not the default value ('labeled_partitions'), "
        "so the taskmodule LabeledSpanExtractionByTokenClassificationTaskModule can not request the usual document type "
        "(TextDocumentWithLabeledSpansAndLabeledPartitions) for auto-conversion because this has "
        "the bespoken default value as layer name(s) instead of the provided one(s)."
    )


def test_document_type_with_non_default_span_and_partition_annotation(caplog):
    with caplog.at_level(logging.WARNING):
        taskmodule = LabeledSpanExtractionByTokenClassificationTaskModule(
            tokenizer_name_or_path="bert-base-uncased",
            span_annotation="entities",
            partition_annotation="sentences",
        )
    assert taskmodule.document_type is None
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert (
        caplog.records[0].message
        == "span_annotation=entities is not the default value ('labeled_spans') and "
        "partition_annotation=sentences is not the default value ('labeled_partitions'), "
        "so the taskmodule LabeledSpanExtractionByTokenClassificationTaskModule can not request the usual document "
        "type (TextDocumentWithLabeledSpansAndLabeledPartitions) for auto-conversion because "
        "this has the bespoken default value as layer name(s) instead of the provided one(s)."
    )


def test_configure_model_metric(documents):
    taskmodule = LabeledSpanExtractionByTokenClassificationTaskModule(
        tokenizer_name_or_path="bert-base-uncased",
        span_annotation="entities",
        labels=["LOC", "PER"],
    )
    taskmodule.post_prepare()

    metric = taskmodule.configure_model_metric(stage="test")
    values = metric.compute()
    assert values == {
        "token/macro/f1": tensor(0.0),
        "token/micro/f1": tensor(0.0),
        "token/macro/precision": tensor(0.0),
        "token/macro/recall": tensor(0.0),
        "token/micro/precision": tensor(0.0),
        "token/micro/recall": tensor(0.0),
    }

    batch = taskmodule.collate(taskmodule.encode(documents, encode_target=True))
    targets = batch[1]
    metric.update(targets, targets)
    values = metric.compute()
    assert values == {
        "span/LOC/f1": tensor(1.0),
        "span/LOC/precision": tensor(1.0),
        "span/LOC/recall": tensor(1.0),
        "span/PER/f1": tensor(1.0),
        "span/PER/precision": tensor(1.0),
        "span/PER/recall": tensor(1.0),
        "span/macro/f1": tensor(1.0),
        "span/macro/precision": tensor(1.0),
        "span/macro/recall": tensor(1.0),
        "span/micro/f1": tensor(1.0),
        "span/micro/precision": tensor(1.0),
        "span/micro/recall": tensor(1.0),
        "token/macro/f1": tensor(1.0),
        "token/micro/f1": tensor(1.0),
        "token/macro/precision": tensor(1.0),
        "token/macro/recall": tensor(1.0),
        "token/micro/precision": tensor(1.0),
        "token/micro/recall": tensor(1.0),
    }

    target_labels = targets["labels"]
    predicted_labels = torch.ones_like(target_labels)
    # we need to set the same padding as in the targets
    predicted_labels[target_labels == taskmodule.label_pad_id] = taskmodule.label_pad_id
    prediction = {"labels": predicted_labels}
    metric.update(prediction, targets)
    values = metric.compute()
    values_converted = {k: v.item() for k, v in values.items()}
    assert values_converted == {
        "token/macro/f1": 0.5434783101081848,
        "token/micro/f1": 0.5249999761581421,
        "token/macro/precision": 0.773809552192688,
        "token/macro/recall": 0.625,
        "token/micro/precision": 0.5249999761581421,
        "token/micro/recall": 0.5249999761581421,
        "span/LOC/recall": 0.0476190485060215,
        "span/LOC/precision": 0.5,
        "span/LOC/f1": 0.08695652335882187,
        "span/macro/f1": 0.37681159377098083,
        "span/macro/precision": 0.5,
        "span/macro/recall": 0.523809552192688,
        "span/micro/recall": 0.1304347813129425,
        "span/micro/precision": 0.5,
        "span/micro/f1": 0.2068965584039688,
        "span/PER/recall": 1.0,
        "span/PER/precision": 0.5,
        "span/PER/f1": 0.6666666865348816,
    }

    # ensure that the metric can be pickled
    pickle.dumps(metric)
