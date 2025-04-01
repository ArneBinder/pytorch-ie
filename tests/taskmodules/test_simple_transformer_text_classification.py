from copy import copy
from dataclasses import dataclass
from typing import Any, Dict

import pytest
import torch
from pie_core import AnnotationLayer, Document, annotation_field
from transformers import BatchEncoding

from pytorch_ie.annotations import Label
from pytorch_ie.taskmodules import SimpleTransformerTextClassificationTaskModule


def _config_to_str(cfg: Dict[str, Any]) -> str:
    # Converts a configuration dictionary to a string representation
    result = "-".join([f"{k}={cfg[k]}" for k in sorted(cfg)])
    return result


CONFIGS = [
    {"max_length": 16},
    {"max_length": 8},
]

CONFIGS_DICT = {_config_to_str(cfg): cfg for cfg in CONFIGS}


@pytest.fixture(scope="module", params=CONFIGS_DICT.keys())
def config(request):
    """
    - Provides taskmodule configuration for testing.
    - Yields config dictionaries from the CONFIGS list to produce clean test case identifiers.

    """
    return CONFIGS_DICT[request.param]


@pytest.fixture(scope="module")
def unprepared_taskmodule(config):
    """
    - Prepares a task module with the specified tokenizer and configuration.
    - Sets up the task module with an unprepared state for testing purposes.

    """
    return SimpleTransformerTextClassificationTaskModule(
        tokenizer_name_or_path="bert-base-uncased", **config
    )


def test_taskmodule(unprepared_taskmodule):
    assert unprepared_taskmodule is not None
    assert not unprepared_taskmodule.is_prepared


@dataclass
class ExampleDocument(Document):
    text: str
    label: AnnotationLayer[Label] = annotation_field()


@pytest.fixture(scope="module")
def documents():
    """
    - Creates example documents with predefined texts.
    - Assigns labels to the documents for testing purposes.

    """
    doc1 = ExampleDocument(text="May your code be bug-free and your algorithms optimized!")
    doc2 = ExampleDocument(
        text="A cascading failure occurred, resulting in a complete system crash and irreversible data loss."
    )
    # assign label
    label1 = Label(label="Positive")
    label2 = Label(label="Negative")
    # add label
    doc1.label.append(label1)
    doc2.label.append(label2)
    return [doc1, doc2]


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
    assert taskmodule.label_to_id == {"O": 0, "Negative": 1, "Positive": 2}
    assert taskmodule.id_to_label == {0: "O", 1: "Negative", 2: "Positive"}


@pytest.fixture(scope="module")
def task_encoding_without_targets(taskmodule, documents):
    """
    - Generates input encodings for a specific task from a document, but without associated targets.

    """
    return taskmodule.encode_input(documents[0])


def test_encode_input(task_encoding_without_targets, documents, taskmodule):
    assert task_encoding_without_targets is not None
    assert task_encoding_without_targets.document == documents[0]
    assert not task_encoding_without_targets.has_targets
    assert set(task_encoding_without_targets.inputs) == {
        "token_type_ids",
        "input_ids",
        "attention_mask",
    }
    if taskmodule.max_length == 8:
        expected_input_tokens = ["[CLS]", "may", "your", "code", "be", "bug", "-", "[SEP]"]
        input_tokens = taskmodule.tokenizer.convert_ids_to_tokens(
            task_encoding_without_targets.inputs["input_ids"]
        )
        assert input_tokens == expected_input_tokens
        assert task_encoding_without_targets.inputs["attention_mask"] == [1] * len(
            expected_input_tokens
        )
    else:
        expected_input_tokens = [
            "[CLS]",
            "may",
            "your",
            "code",
            "be",
            "bug",
            "-",
            "free",
            "and",
            "your",
            "algorithms",
            "opt",
            "##imi",
            "##zed",
            "!",
            "[SEP]",
        ]
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
    """
    - Encodes the target for a given task encoding.
    - Generates encoded targets for a specific task encoding.

    """
    return taskmodule.encode_target(task_encoding_without_targets)


def test_target(target, taskmodule):
    expected_label = "Positive"
    label_tokens = taskmodule.id_to_label[target]
    assert label_tokens == expected_label


@pytest.fixture(scope="module")
def task_encoding(task_encoding_without_targets, target):
    """
    - Combines the task encoding with the associated target.
    - Creates a new task encoding by copying the original and including the target.

    """
    result = copy(task_encoding_without_targets)
    result.targets = target
    return result


def test_task_encoding(task_encoding):
    assert task_encoding is not None


@pytest.fixture(scope="module")
def batch(taskmodule, task_encoding_without_targets):
    """
    - Collates a list of task encodings into a batch.
    - Prepares a batch of task encodings for efficient processing.

    """
    task_encodings = [task_encoding_without_targets, task_encoding_without_targets]
    return taskmodule.collate(task_encodings)


def test_collate(batch, taskmodule):
    assert batch is not None
    assert len(batch) == 2
    batch_encoding, _ = batch
    if taskmodule.max_length == 8:
        input_ids_expected = torch.tensor(
            [
                [101, 2089, 2115, 3642, 2022, 11829, 1011, 102],
                [101, 2089, 2115, 3642, 2022, 11829, 1011, 102],
            ],
            dtype=torch.int64,
        )
        attention_mask_expected = torch.tensor(
            [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.int64
        )
        token_type_ids_expected = torch.tensor(
            [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.int64
        )
        encoding_expected = BatchEncoding(
            data={
                "input_ids": input_ids_expected,
                "attention_mask": attention_mask_expected,
                "token_type_ids": token_type_ids_expected,
            }
        )
    else:
        input_ids_expected = torch.tensor(
            [
                [
                    101,
                    2089,
                    2115,
                    3642,
                    2022,
                    11829,
                    1011,
                    2489,
                    1998,
                    2115,
                    13792,
                    23569,
                    27605,
                    5422,
                    999,
                    102,
                ],
                [
                    101,
                    2089,
                    2115,
                    3642,
                    2022,
                    11829,
                    1011,
                    2489,
                    1998,
                    2115,
                    13792,
                    23569,
                    27605,
                    5422,
                    999,
                    102,
                ],
            ],
            dtype=torch.int64,
        )
        attention_mask_expected = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ],
            dtype=torch.int64,
        )
        token_type_ids_expected = torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=torch.int64,
        )
        encoding_expected = BatchEncoding(
            data={
                "input_ids": input_ids_expected,
                "attention_mask": attention_mask_expected,
                "token_type_ids": token_type_ids_expected,
            }
        )
    assert set(batch_encoding.data) == set(encoding_expected.data)
    torch.testing.assert_close(batch_encoding.input_ids, encoding_expected.input_ids)
    torch.testing.assert_close(batch_encoding.attention_mask, encoding_expected.attention_mask)
    torch.testing.assert_close(batch_encoding.token_type_ids, encoding_expected.token_type_ids)


# This is not used, but can be used to create a batch of task encodings with targets for the unbatched_outputs fixture.
@pytest.fixture(scope="module")
def model_predict_output(batch, taskmodule):
    """
    - Initializes and predicts the model outputs for the given batch.
    - Creates an instance of TransformerTextClassificationModel and passes the batch through it.
    - Returns the model's output predictions.

    """
    from pytorch_ie import TransformerTextClassificationModel

    model = TransformerTextClassificationModel(
        model_name_or_path="prajjwal1/bert-tiny",
        num_classes=len(taskmodule.label_to_id),
        t_total=1000,
        tokenizer_vocab_size=len(taskmodule.tokenizer),
    )
    input, target = batch
    result = model(input)
    return result


@pytest.fixture(scope="module")
def unbatched_outputs(taskmodule):
    """
    - Converts model outputs from batched to unbatched format.
    - Helps in further processing of model outputs for individual task encodings.
    - Model output can be created with model_predict_output fixture above.

    """
    model_output = {"logits": torch.tensor([[0.0513, 0.7510, -0.3345], [0.7510, 0.0513, -0.3345]])}
    return taskmodule.unbatch_output(model_output)


def test_unpatch_output(unbatched_outputs):
    assert unbatched_outputs is not None
    assert unbatched_outputs == [
        {"label": "Negative", "probability": 0.5451174378395081},
        {"label": "O", "probability": 0.5451174378395081},
    ]


@pytest.fixture(scope="module")
def annotations_from_output(taskmodule, task_encoding_without_targets, unbatched_outputs):
    """
    Converts the inputs (task_encoding_without_targets) and the respective model outputs (unbatched_outputs)
    into human-readable  annotations.

    """
    task_encodings = [task_encoding_without_targets, task_encoding_without_targets]
    assert len(task_encodings) == len(unbatched_outputs)
    named_annotations = []
    for task_encoding, task_output in zip(task_encodings, unbatched_outputs):
        annotations = taskmodule.create_annotations_from_output(task_encoding, task_output)
        named_annotations.extend(annotations)
    return named_annotations


def test_annotations_from_output(annotations_from_output):
    assert annotations_from_output is not None
    assert len(annotations_from_output) == 2
    assert annotations_from_output[0] == (
        "label",
        Label(label="Negative", score=0.5451174378395081),
    )
    assert annotations_from_output[1] == ("label", Label(label="O", score=0.5451174378395081))
