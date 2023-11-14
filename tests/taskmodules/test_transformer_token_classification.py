from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pytest
import torch
from transformers import BatchEncoding

from pytorch_ie import AnnotationLayer, Document, annotation_field
from pytorch_ie.annotations import LabeledSpan, Span
from pytorch_ie.taskmodules import TransformerTokenClassificationTaskModule


def _config_to_str(cfg: Dict[str, Any]) -> str:
    # Converts a configuration dictionary to a string representation
    result = "-".join([f"{k}={cfg[k]}" for k in sorted(cfg)])
    return result


CONFIGS: List[Dict[str, Any]] = [
    {},
    {"max_window": 8},
    {"max_window": 8, "window_overlap": 2},
    {"partition_annotation": "sentences"},
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
    return TransformerTokenClassificationTaskModule(
        tokenizer_name_or_path="bert-base-uncased", entity_annotation="entities", **config
    )


@dataclass
class ExampleDocument(Document):
    text: str
    entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
    sentences: AnnotationLayer[Span] = annotation_field(target="text")


@pytest.fixture(scope="module")
def documents():
    """
    - Creates example documents with predefined texts.
    - Assigns labels to the documents for testing purposes.

    """
    doc1 = ExampleDocument(text="Mount Everest is the highest peak in the world.")
    doc2 = ExampleDocument(text="Alice loves reading books. Bob enjoys playing soccer.")
    entity_1 = LabeledSpan(start=0, end=13, label="head")
    entity_2 = LabeledSpan(start=0, end=5, label="head")
    sentence_2 = Span(start=27, end=53)
    doc1.entities.append(entity_1)
    doc2.entities.append(entity_2)
    doc2.sentences.append(sentence_2)
    assert str(entity_1) == "Mount Everest"
    assert str(entity_2) == "Alice"
    assert str(sentence_2) == "Bob enjoys playing soccer."
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
    assert taskmodule.label_to_id == {"O": 0, "B-head": 1, "I-head": 2}
    assert taskmodule.id_to_label == {0: "O", 1: "B-head", 2: "I-head"}


def test_config(taskmodule):
    config = taskmodule._config()
    assert config["taskmodule_type"] == "TransformerTokenClassificationTaskModule"
    assert "label_to_id" in config
    assert config["label_to_id"] == {"O": 0, "B-head": 1, "I-head": 2}


@pytest.fixture(scope="module")
def task_encodings_without_targets(taskmodule, documents):
    """
    - Generates input encodings for all the documents, but without associated targets.
    """
    task_encodings = []
    for i in range(len(documents)):
        task_encodings.append(taskmodule.encode_input(documents[i]))
    return task_encodings


def test_encode_inputs(task_encodings_without_targets, documents, taskmodule, config):
    """
    - Test the encoding of inputs for the model based on the given configuration.

    - Parameters:
        task_encodings_without_targets (list): List of task encodings without targets.
        documents (list): List of documents for testing.
        taskmodule (object): The task module to test.
        config (dict): The configuration to check different cases.
    """
    assert task_encodings_without_targets is not None
    # If config is empty
    if config == {}:
        # Check first document encoding
        assert task_encodings_without_targets[0][0].document == documents[0]
        assert not task_encodings_without_targets[0][0].has_targets
        assert set(task_encodings_without_targets[0][0].inputs) == {
            "token_type_ids",
            "input_ids",
            "attention_mask",
        }
        expected_input_tokens = [
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
        ]

        input_tokens = taskmodule.tokenizer.convert_ids_to_tokens(
            task_encodings_without_targets[0][0].inputs["input_ids"]
        )
        assert input_tokens == expected_input_tokens
        assert task_encodings_without_targets[0][0].inputs["attention_mask"] == [1] * len(
            expected_input_tokens
        )
    # If config has the specified values (max_window=8, window_overlap=2)
    elif config == {"max_window": 8, "window_overlap": 2}:
        assert len(task_encodings_without_targets[0]) == 4
        # Iterate over each part of task_encodings_without_targets[0]
        for i in range(0, len(task_encodings_without_targets[0])):
            assert task_encodings_without_targets[0][i].document == documents[0]
            assert not task_encodings_without_targets[0][i].has_targets
            if i == 0:
                assert set(task_encodings_without_targets[0][i].inputs) == {"input_ids"}
                expected_input_tokens = [
                    "[CLS]",
                    "mount",
                    "everest",
                    "is",
                    "the",
                    "highest",
                    "peak",
                    "[SEP]",
                ]
                input_tokens = taskmodule.tokenizer.convert_ids_to_tokens(
                    task_encodings_without_targets[0][i].inputs["input_ids"]
                )
                assert input_tokens == expected_input_tokens
            elif i == 1:
                assert set(task_encodings_without_targets[0][i].inputs) == {"input_ids"}
                expected_input_tokens = [
                    "[CLS]",
                    "is",
                    "the",
                    "highest",
                    "peak",
                    "in",
                    "the",
                    "[SEP]",
                ]
                input_tokens = taskmodule.tokenizer.convert_ids_to_tokens(
                    task_encodings_without_targets[0][i].inputs["input_ids"]
                )
                assert input_tokens == expected_input_tokens
            elif i == 2:
                assert set(task_encodings_without_targets[0][i].inputs) == {"input_ids"}
                expected_input_tokens = [
                    "[CLS]",
                    "highest",
                    "peak",
                    "in",
                    "the",
                    "world",
                    ".",
                    "[SEP]",
                ]
                input_tokens = taskmodule.tokenizer.convert_ids_to_tokens(
                    task_encodings_without_targets[0][i].inputs["input_ids"]
                )
                assert input_tokens == expected_input_tokens
            else:
                assert set(task_encodings_without_targets[0][i].inputs) == {"input_ids"}
                expected_input_tokens = ["[CLS]", "in", "the", "world", ".", "[SEP]"]
                input_tokens = taskmodule.tokenizer.convert_ids_to_tokens(
                    task_encodings_without_targets[0][i].inputs["input_ids"]
                )
                assert input_tokens == expected_input_tokens

    # If config has the specified value (max_window=8)
    elif config == {"max_window": 8}:
        assert len(task_encodings_without_targets[0]) == 2
        # Iterate over each part of task_encodings_without_targets[0]
        for i in range(0, len(task_encodings_without_targets[0])):
            assert task_encodings_without_targets[0][i].document == documents[0]
            assert not task_encodings_without_targets[0][i].has_targets
            if i == 0:
                assert set(task_encodings_without_targets[0][i].inputs) == {"input_ids"}
                expected_input_tokens = [
                    "[CLS]",
                    "mount",
                    "everest",
                    "is",
                    "the",
                    "highest",
                    "peak",
                    "[SEP]",
                ]
                input_tokens = taskmodule.tokenizer.convert_ids_to_tokens(
                    task_encodings_without_targets[0][i].inputs["input_ids"]
                )
                assert input_tokens == expected_input_tokens
            else:
                assert set(task_encodings_without_targets[0][i].inputs) == {"input_ids"}
                expected_input_tokens = ["[CLS]", "in", "the", "world", ".", "[SEP]"]
                input_tokens = taskmodule.tokenizer.convert_ids_to_tokens(
                    task_encodings_without_targets[0][i].inputs["input_ids"]
                )
                assert input_tokens == expected_input_tokens

    # If config has the specified value (partition_annotation=sentences)
    elif config == {"partition_annotation": "sentences"}:
        assert task_encodings_without_targets[1][0].document == documents[1]
        assert not task_encodings_without_targets[1][0].has_targets
        assert set(task_encodings_without_targets[1][0].inputs) == {
            "token_type_ids",
            "input_ids",
            "attention_mask",
        }
        expected_input_tokens = ["[CLS]", "bob", "enjoys", "playing", "soccer", ".", "[SEP]"]
        input_tokens = taskmodule.tokenizer.convert_ids_to_tokens(
            task_encodings_without_targets[1][0].inputs["input_ids"]
        )
        assert input_tokens == expected_input_tokens

    else:
        raise ValueError(f"unknown config: {config}")


@pytest.fixture(scope="module")
def targets(taskmodule, task_encodings_without_targets, config):
    """
    - Encodes the target for a given task encoding.
    - Generates encoded targets for a specific task encoding.
    - For config value (partition_annotation=sentences), taking the second documents as first document don't have sentences entity.
    """
    targets = []
    # Here ctr represents document index
    ctr = 0
    if config != {"partition_annotation": "sentences"}:
        ctr = 0
    else:
        ctr = 1
    for i in range(len(task_encodings_without_targets[ctr])):
        targets.append(taskmodule.encode_target(task_encodings_without_targets[ctr][i]))
    return targets


def test_target(targets, taskmodule, config):
    labels_tokens = []

    # If config is empty
    if config == {}:
        assert len(targets) == 1
        """
        expected_input_tokens = ["[CLS]","mount","everest","is","the","highest","peak","in","the","world",".","[SEP]",]
        """
        expected_labels = [
            ["<pad>", "B-head", "I-head", "O", "O", "O", "O", "O", "O", "O", "O", "<pad>"]
        ]
        labels_tokens.append(
            [taskmodule.id_to_label[x] if x != -100 else "<pad>" for x in targets[0]]
        )

    # If config has the specified values (max_window=8, window_overlap=2)
    elif config == {"max_window": 8, "window_overlap": 2}:
        assert len(targets) == 4
        """
        expected_input_tokens = [
            ["[CLS]","mount","everest","is","the","highest","peak","[SEP]"],
            ["[CLS]","is","the","highest","peak","in","the","[SEP]"],
            ["[CLS]","highest","peak","in","the","world",".","[SEP]"],
            ["[CLS]", "in", "the", "world", ".", "[SEP]"],
        ]
        """
        expected_labels = [
            ["<pad>", "B-head", "I-head", "O", "O", "<pad>", "<pad>", "<pad>"],
            ["<pad>", "<pad>", "<pad>", "O", "O", "<pad>", "<pad>", "<pad>"],
            ["<pad>", "<pad>", "<pad>", "O", "O", "O", "O", "<pad>"],
            ["<pad>", "<pad>", "<pad>", "O", "O", "<pad>"],
        ]
        for i in range(len(targets)):
            labels_tokens.append(
                [taskmodule.id_to_label[x] if x != -100 else "<pad>" for x in targets[i]]
            )

    # If config has the specified value (max_window=8)
    elif config == {"max_window": 8}:
        assert len(targets) == 2
        """
        expected_input_tokens = [
            ["[CLS]","mount","everest","is","the","highest","peak","[SEP]"],
            ["[CLS]", "in", "the", "world", ".", "[SEP]"]
        ]
        """
        expected_labels = [
            ["<pad>", "B-head", "I-head", "O", "O", "O", "O", "<pad>"],
            ["<pad>", "O", "O", "O", "O", "<pad>"],
        ]
        for i in range(len(targets)):
            labels_tokens.append(
                [taskmodule.id_to_label[x] if x != -100 else "<pad>" for x in targets[i]]
            )

    # If config has the specified value (partition_annotation=sentences)
    elif config == {"partition_annotation": "sentences"}:
        assert len(targets) == 1
        """
        expected_input_tokens = ["[CLS]", "bob", "enjoys", "playing", "soccer", ".", "[SEP]"]
        """
        expected_labels = [["<pad>", "O", "O", "O", "O", "O", "<pad>"]]
        labels_tokens.append(
            [taskmodule.id_to_label[x] if x != -100 else "<pad>" for x in targets[0]]
        )

    else:
        raise ValueError(f"unknown config: {config}")

    assert expected_labels == labels_tokens


@pytest.fixture(scope="module")
def batch(taskmodule, task_encodings_without_targets, config):
    """
    - Collates a list of task encodings into a batch.
    - Prepares a batch of task encodings for efficient processing.
    - To maintain the same batch size for all configs, the first document is duplicated in task_encodings_without_targets when using the "partition_annotation=sentences" config, as it initially contains no values for the first document.
    """
    if config != {"partition_annotation": "sentences"}:
        task_encodings = [
            task_encodings_without_targets[0][0],
            task_encodings_without_targets[1][0],
        ]
    else:
        task_encodings = [
            task_encodings_without_targets[1][0],
            task_encodings_without_targets[1][0],
        ]
    return taskmodule.collate(task_encodings)


def test_collate(batch, config):
    """
    - Test the collate function that creates batch encodings based on the specified configuration.

    - Parameters:
        batch (tuple): A tuple containing the batch encoding and other metadata.
        config (dict): A dictionary containing configuration settings for the collation.
    """
    assert batch is not None
    assert len(batch) == 2
    batch_encoding, _ = batch

    # If config is empty
    if config == {}:
        input_ids_expected = torch.tensor(
            [
                [101, 4057, 23914, 2003, 1996, 3284, 4672, 1999, 1996, 2088, 1012, 102],
                [101, 5650, 7459, 3752, 2808, 1012, 3960, 15646, 2652, 4715, 1012, 102],
            ],
            dtype=torch.int64,
        )
        token_type_ids_expected = torch.tensor(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
            dtype=torch.int64,
        )
        attention_mask_expected = torch.tensor(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
            dtype=torch.int64,
        )

        encoding_expected = BatchEncoding(
            data={
                "input_ids": input_ids_expected,
                "attention_mask": attention_mask_expected,
                "token_type_ids": token_type_ids_expected,
            }
        )
        torch.testing.assert_close(batch_encoding.input_ids, encoding_expected.input_ids)
        torch.testing.assert_close(batch_encoding.attention_mask, encoding_expected.attention_mask)
        torch.testing.assert_close(batch_encoding.token_type_ids, encoding_expected.token_type_ids)

    # If config has the specified values (max_window=8, window_overlap=2)
    elif config == {"max_window": 8, "window_overlap": 2}:
        input_ids_expected = torch.tensor(
            [
                [101, 4057, 23914, 2003, 1996, 3284, 4672, 102],
                [101, 5650, 7459, 3752, 2808, 1012, 3960, 102],
            ],
            dtype=torch.int64,
        )
        attention_mask_expected = torch.tensor(
            [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.int64
        )
        encoding_expected = BatchEncoding(
            data={
                "input_ids": input_ids_expected,
                "attention_mask": attention_mask_expected,
            }
        )
        torch.testing.assert_close(batch_encoding.input_ids, encoding_expected.input_ids)
        torch.testing.assert_close(batch_encoding.attention_mask, encoding_expected.attention_mask)

    # If config has the specified values (max_window=8)
    elif config == {"max_window": 8}:
        input_ids_expected = torch.tensor(
            [
                [101, 4057, 23914, 2003, 1996, 3284, 4672, 102],
                [101, 5650, 7459, 3752, 2808, 1012, 3960, 102],
            ],
            dtype=torch.int64,
        )
        attention_mask_expected = torch.tensor(
            [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.int64
        )
        encoding_expected = BatchEncoding(
            data={
                "input_ids": input_ids_expected,
                "attention_mask": attention_mask_expected,
            }
        )
        torch.testing.assert_close(batch_encoding.input_ids, encoding_expected.input_ids)
        torch.testing.assert_close(batch_encoding.attention_mask, encoding_expected.attention_mask)

    # If config has the specified value (partition_annotation=sentences)
    elif config == {"partition_annotation": "sentences"}:
        input_ids_expected = torch.tensor(
            [[101, 3960, 15646, 2652, 4715, 1012, 102], [101, 3960, 15646, 2652, 4715, 1012, 102]],
            dtype=torch.int64,
        )
        token_type_ids_expected = torch.tensor(
            [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], dtype=torch.int64
        )
        attention_mask_expected = torch.tensor(
            [[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]], dtype=torch.int64
        )
        encoding_expected = BatchEncoding(
            data={
                "input_ids": input_ids_expected,
                "attention_mask": attention_mask_expected,
                "token_type_ids": token_type_ids_expected,
            }
        )

        torch.testing.assert_close(batch_encoding.input_ids, encoding_expected.input_ids)
        torch.testing.assert_close(batch_encoding.attention_mask, encoding_expected.attention_mask)
        torch.testing.assert_close(batch_encoding.token_type_ids, encoding_expected.token_type_ids)

    assert set(batch_encoding.data) == set(encoding_expected.data)


# This is not used, but can be used to create a batch of task encodings with targets for the unbatched_outputs fixture.
@pytest.fixture(scope="module")
def model_predict_output(batch, taskmodule):
    """
    - Initializes and predicts the model outputs for the given batch.
    - Creates an instance of TransformerTextClassificationModel and passes the batch through it.
    - Returns the model's output predictions.

    """
    from pytorch_ie import TransformerTokenClassificationModel

    model = TransformerTokenClassificationModel(
        model_name_or_path="prajjwal1/bert-tiny",
        num_classes=len(taskmodule.label_to_id),
    )
    input, target = batch
    result = model(input)
    return result


@pytest.fixture(scope="module")
def unbatched_outputs(taskmodule, config):
    # If config is empty
    if config == {}:
        model_output = {
            "logits": torch.tensor(
                [
                    [
                        [-0.0659, 0.0170, -0.2684],
                        [-0.0418, 0.1595, -0.2855],
                        [0.0561, 0.1375, -0.2456],
                        [-0.1719, 0.2413, -0.2220],
                        [-0.2429, 0.1623, -0.2379],
                        [-0.2246, 0.1382, -0.2564],
                        [-0.1231, 0.1595, -0.3846],
                        [-0.2681, 0.1534, -0.2445],
                        [-0.2461, 0.2414, -0.3293],
                        [-0.1729, 0.2220, -0.1880],
                        [-0.2740, 0.2431, -0.1882],
                        [-0.2420, 0.1079, -0.2696],
                    ],
                    [
                        [0.0140, -0.1751, 0.1674],
                        [0.0297, -0.0988, 0.0006],
                        [-0.1173, -0.1797, 0.0936],
                        [-0.2464, -0.2545, 0.1067],
                        [-0.3522, -0.1276, 0.0111],
                        [-0.1681, 0.0503, 0.0019],
                        [0.0713, 0.1196, -0.1907],
                        [-0.1181, -0.0307, 0.0633],
                        [-0.3371, 0.1819, 0.0052],
                        [-0.2783, -0.0957, -0.0271],
                        [-0.2880, 0.0547, 0.0221],
                        [-0.2033, -0.0376, 0.0898],
                    ],
                ]
            )
        }

    # If config has the specified values (max_window=8, window_overlap=2)
    elif config == {"max_window": 8, "window_overlap": 2}:
        model_output = {
            "logits": torch.tensor(
                [
                    [
                        [-0.2204, 0.2539, 0.0036],
                        [-0.2380, 0.1804, 0.0673],
                        [-0.3890, 0.2565, 0.1223],
                        [-0.2411, 0.3255, -0.1082],
                        [-0.2355, 0.4625, -0.1610],
                        [-0.1030, 0.1193, -0.1866],
                        [-0.1501, 0.2016, -0.1718],
                        [-0.2469, 0.2522, -0.1166],
                    ],
                    [
                        [-0.2366, 0.1041, -0.2780],
                        [-0.3588, 0.0749, -0.1663],
                        [-0.4543, 0.0175, -0.3157],
                        [-0.4051, 0.0334, -0.1502],
                        [-0.4849, 0.3890, -0.2533],
                        [-0.6248, 0.3296, -0.0093],
                        [-0.5428, 0.3440, 0.0266],
                        [-0.3864, 0.0836, -0.0438],
                    ],
                ]
            )
        }
    # If config has the specified values (max_window=8)
    elif config == {"max_window": 8}:
        model_output = {
            "logits": torch.tensor(
                [
                    [
                        [-0.1508, 0.3434, 0.3668],
                        [-0.1872, 0.1007, 0.2948],
                        [-0.0732, 0.0601, 0.2213],
                        [-0.1128, 0.0704, 0.2546],
                        [-0.0987, 0.2763, 0.2852],
                        [0.1105, 0.2054, 0.4415],
                        [-0.0376, 0.3338, 0.3140],
                        [-0.0937, 0.2559, 0.0492],
                    ],
                    [
                        [-0.3258, 0.1260, 0.1610],
                        [-0.3489, -0.0896, 0.0903],
                        [-0.2561, -0.2279, 0.0045],
                        [-0.2420, -0.1238, 0.0231],
                        [-0.3167, -0.0356, -0.0050],
                        [-0.2999, 0.0668, -0.1417],
                        [-0.2031, -0.1222, 0.0272],
                        [-0.3968, -0.2068, -0.2290],
                    ],
                ]
            )
        }

    # If config has the specified value (partition_annotation=sentences)
    elif config == {"partition_annotation": "sentences"}:
        model_output = {
            "logits": torch.tensor(
                [
                    [
                        [0.2960, -0.0264, -0.1626],
                        [0.0915, 0.1708, 0.0648],
                        [0.2399, -0.1459, -0.1110],
                        [0.3249, 0.2534, -0.1120],
                        [0.2190, 0.1073, 0.0196],
                        [0.1986, 0.2853, 0.3358],
                        [0.1038, 0.1871, -0.0320],
                    ],
                    [
                        [0.2960, -0.0264, -0.1626],
                        [0.0915, 0.1708, 0.0648],
                        [0.2399, -0.1459, -0.1110],
                        [0.3249, 0.2534, -0.1120],
                        [0.2190, 0.1073, 0.0196],
                        [0.1986, 0.2853, 0.3358],
                        [0.1038, 0.1871, -0.0320],
                    ],
                ]
            )
        }

    else:
        raise ValueError(f"unknown config: {config}")

    return taskmodule.unbatch_output(model_output)


def test_unbatch_output(unbatched_outputs, config):
    """
    - Test the unbatched outputs generated by the model.

    - Parameters:
        unbatched_outputs (list): List of unbatched outputs from the model.
        config (dict): The configuration to check different cases.

    - Perform assertions for each unbatched output based on the given configuration.
    """
    assert unbatched_outputs is not None
    assert len(unbatched_outputs) == 2

    # Based on the config, perform assertions for each unbatched output
    if config == {}:
        # Assertions for the first unbatched output
        assert unbatched_outputs[0]["tags"] == [
            "B-head",
            "B-head",
            "B-head",
            "B-head",
            "B-head",
            "B-head",
            "B-head",
            "B-head",
            "B-head",
            "B-head",
            "B-head",
            "B-head",
        ]
        assert np.all(
            unbatched_outputs[0]["probabilities"]
            == np.array(
                [
                    [0.344457, 0.37422952, 0.2813134],
                    [0.33258897, 0.4067535, 0.26065755],
                    [0.35406193, 0.38408807, 0.26185],
                    [0.2887852, 0.43654135, 0.2746735],
                    [0.28533804, 0.4278936, 0.2867683],
                    [0.29359534, 0.42199877, 0.2844059],
                    [0.32294837, 0.42841503, 0.2486366],
                    [0.28183886, 0.4295918, 0.28856936],
                    [0.28181657, 0.45886514, 0.25931832],
                    [0.2882468, 0.42782623, 0.283927],
                    [0.2654812, 0.44525358, 0.28926522],
                    [0.29483712, 0.41835198, 0.2868109],
                ],
                dtype=np.float32,
            )
        )
        # Assertions for the second unbatched output
        assert unbatched_outputs[1]["tags"] == [
            "I-head",
            "O",
            "I-head",
            "I-head",
            "I-head",
            "B-head",
            "B-head",
            "I-head",
            "B-head",
            "I-head",
            "B-head",
            "I-head",
        ]
        assert np.all(
            unbatched_outputs[1]["probabilities"]
            == np.array(
                [
                    [0.3340577, 0.2765008, 0.38944152],
                    [0.35078698, 0.30848682, 0.34072617],
                    [0.3150305, 0.29597336, 0.38899615],
                    [0.29279095, 0.2904289, 0.41678014],
                    [0.27101088, 0.3392573, 0.38973182],
                    [0.2915971, 0.36277145, 0.34563145],
                    [0.35473615, 0.37229043, 0.2729734],
                    [0.3039303, 0.33168924, 0.36438042],
                    [0.24458675, 0.41099048, 0.34442282],
                    [0.28686982, 0.34433967, 0.36879045],
                    [0.26508972, 0.3734441, 0.36146614],
                    [0.2840267, 0.33521372, 0.38075963],
                ],
                dtype=np.float32,
            )
        )

    elif config == {"max_window": 8, "window_overlap": 2}:
        # Assertions for the first unbatched output
        assert unbatched_outputs[0]["tags"] == [
            "B-head",
            "B-head",
            "B-head",
            "B-head",
            "B-head",
            "B-head",
            "B-head",
            "B-head",
        ]
        assert np.all(
            unbatched_outputs[0]["probabilities"]
            == np.array(
                [
                    [0.25920436, 0.4165126, 0.32428306],
                    [0.25796065, 0.3919785, 0.35006085],
                    [0.21860804, 0.41687244, 0.3645196],
                    [0.25612125, 0.45135355, 0.29252523],
                    [0.24467377, 0.49172804, 0.26359814],
                    [0.31558236, 0.3941453, 0.29027236],
                    [0.2941163, 0.41808102, 0.28780273],
                    [0.26410252, 0.43503976, 0.30085772],
                ],
                dtype=np.float32,
            )
        )
        # Assertions for the second unbatched output
        assert unbatched_outputs[1]["tags"] == [
            "B-head",
            "B-head",
            "B-head",
            "B-head",
            "B-head",
            "B-head",
            "B-head",
            "B-head",
        ]
        assert np.all(
            unbatched_outputs[1]["probabilities"]
            == np.array(
                [
                    [0.29714355, 0.41776344, 0.28509298],
                    [0.26629514, 0.4108816, 0.3228233],
                    [0.26655713, 0.4272582, 0.3061847],
                    [0.26036835, 0.40366986, 0.3359618],
                    [0.2147373, 0.51456165, 0.2707011],
                    [0.18356392, 0.47673604, 0.33970004],
                    [0.19250922, 0.46728724, 0.34020358],
                    [0.24946368, 0.39914045, 0.35139585],
                ],
                dtype=np.float32,
            )
        )

    elif config == {"max_window": 8}:
        # Assertions for the first unbatched output
        assert unbatched_outputs[0]["tags"] == [
            "I-head",
            "I-head",
            "I-head",
            "I-head",
            "I-head",
            "I-head",
            "B-head",
            "B-head",
        ]
        assert np.all(
            unbatched_outputs[0]["probabilities"]
            == np.array(
                [
                    [0.23163259, 0.37968898, 0.38867846],
                    [0.25297666, 0.33737573, 0.4096476],
                    [0.28694013, 0.3278557, 0.3852042],
                    [0.27434617, 0.32950473, 0.3961491],
                    [0.25490764, 0.3708884, 0.37420404],
                    [0.28637636, 0.31488478, 0.3987389],
                    [0.25832433, 0.374509, 0.36716667],
                    [0.27994624, 0.39710376, 0.32295004],
                ],
                dtype=np.float32,
            )
        )

        # Assertions for the second unbatched output
        assert unbatched_outputs[1]["tags"] == [
            "I-head",
            "I-head",
            "I-head",
            "I-head",
            "I-head",
            "B-head",
            "I-head",
            "B-head",
        ]
        assert np.all(
            unbatched_outputs[1]["probabilities"]
            == np.array(
                [
                    [0.2381951, 0.37423733, 0.38756755],
                    [0.25990984, 0.336849, 0.40324116],
                    [0.30063346, 0.30923197, 0.39013457],
                    [0.29162762, 0.32821786, 0.38015446],
                    [0.2709784, 0.35893422, 0.3700874],
                    [0.27667376, 0.39923054, 0.32409576],
                    [0.29911104, 0.32431486, 0.3765741],
                    [0.29481572, 0.35650578, 0.34867856],
                ],
                dtype=np.float32,
            )
        )

    elif config == {"partition_annotation": "sentences"}:
        # Assertions for the first unbatched output
        assert unbatched_outputs[0]["tags"] == [
            "O",
            "B-head",
            "O",
            "O",
            "O",
            "I-head",
            "B-head",
        ]
        assert np.all(
            unbatched_outputs[0]["probabilities"]
            == np.array(
                [
                    [0.4243444, 0.30739865, 0.26825696],
                    [0.3272056, 0.35420957, 0.31858483],
                    [0.41947, 0.2852004, 0.2953296],
                    [0.38804325, 0.36126682, 0.25068992],
                    [0.36852303, 0.32957476, 0.30190223],
                    [0.3088682, 0.3368422, 0.3542896],
                    [0.33785096, 0.36719933, 0.29494968],
                ],
                dtype=np.float32,
            )
        )

        # Assertions for the second unbatched output
        assert unbatched_outputs[1]["tags"] == [
            "O",
            "B-head",
            "O",
            "O",
            "O",
            "I-head",
            "B-head",
        ]
        assert np.all(
            unbatched_outputs[1]["probabilities"]
            == np.array(
                [
                    [0.4243444, 0.30739865, 0.26825696],
                    [0.3272056, 0.35420957, 0.31858483],
                    [0.41947, 0.2852004, 0.2953296],
                    [0.38804325, 0.36126682, 0.25068992],
                    [0.36852303, 0.32957476, 0.30190223],
                    [0.3088682, 0.3368422, 0.3542896],
                    [0.33785096, 0.36719933, 0.29494968],
                ],
                dtype=np.float32,
            )
        )

    else:
        raise ValueError(f"unknown config: {config}")


@pytest.fixture(scope="module")
def annotations_from_output(taskmodule, task_encodings_without_targets, unbatched_outputs, config):
    """
    - Converts the inputs (task_encoding_without_targets) and the respective model outputs (unbatched_outputs)
    into human-readable  annotations.

    """
    if config != {"partition_annotation": "sentences"}:
        task_encodings = [
            task_encodings_without_targets[0][0],
            task_encodings_without_targets[1][0],
        ]
    else:
        task_encodings = [
            task_encodings_without_targets[1][0],
            task_encodings_without_targets[1][0],
        ]
    assert len(task_encodings_without_targets) == len(unbatched_outputs)
    named_annotations = []
    for task_encoding, task_output in zip(task_encodings, unbatched_outputs):
        annotations = taskmodule.create_annotations_from_output(task_encoding, task_output)
        named_annotations.append(list(annotations))
    return named_annotations


def test_annotations_from_output(annotations_from_output, config):
    """
    - Test the annotations generated from the output.

    - Parameters:
        annotations_from_output (list): List of annotations from the model output.
        config (dict): The configuration to check different cases.

    - For each configuration, check the first two entries from annotations_from_output for both documents.
    """
    assert annotations_from_output is not None  # Check that annotations_from_output is not None
    # Sort the annotations in each document by start and end positions
    annotations_from_output = [
        sorted(annotations, key=lambda x: (x[0], x[1].start, x[1].end))
        for annotations in annotations_from_output
    ]
    assert annotations_from_output is not None
    # Check based on the config
    if config == {}:
        # Assertions for the first document
        assert len(annotations_from_output[0]) == 10
        assert annotations_from_output[0][0] == (
            "entities",
            LabeledSpan(start=0, end=5, label="head", score=1.0),
        )
        assert annotations_from_output[0][1] == (
            "entities",
            LabeledSpan(start=6, end=13, label="head", score=1.0),
        )
        # Assertions for the second document
        assert len(annotations_from_output[1]) == 5
        assert annotations_from_output[1][0] == (
            "entities",
            LabeledSpan(start=6, end=25, label="head", score=1.0),
        )
        assert annotations_from_output[1][1] == (
            "entities",
            LabeledSpan(start=25, end=26, label="head", score=1.0),
        )

    elif config == {"max_window": 8, "window_overlap": 2}:
        # Assertions for the first document
        assert len(annotations_from_output[0]) == 4
        assert annotations_from_output[0][0] == (
            "entities",
            LabeledSpan(start=0, end=5, label="head", score=1.0),
        )
        assert annotations_from_output[0][1] == (
            "entities",
            LabeledSpan(start=6, end=13, label="head", score=1.0),
        )
        # Assertions for the second document
        assert len(annotations_from_output[1]) == 4
        assert annotations_from_output[1][0] == (
            "entities",
            LabeledSpan(start=0, end=5, label="head", score=1.0),
        )
        assert annotations_from_output[1][1] == (
            "entities",
            LabeledSpan(start=6, end=11, label="head", score=1.0),
        )

    elif config == {"max_window": 8}:
        # Assertions for the first document
        assert len(annotations_from_output[0]) == 2
        assert annotations_from_output[0][0] == (
            "entities",
            LabeledSpan(start=0, end=28, label="head", score=1.0),
        )
        assert annotations_from_output[0][1] == (
            "entities",
            LabeledSpan(start=29, end=33, label="head", score=1.0),
        )
        # Assertions for the second document
        assert len(annotations_from_output[1]) == 2
        assert annotations_from_output[1][0] == (
            "entities",
            LabeledSpan(start=0, end=25, label="head", score=1.0),
        )
        assert annotations_from_output[1][1] == (
            "entities",
            LabeledSpan(start=25, end=30, label="head", score=1.0),
        )

    elif config == {"partition_annotation": "sentences"}:
        # Assertions for the first document
        assert len(annotations_from_output[0]) == 2
        assert annotations_from_output[0][0] == (
            "entities",
            LabeledSpan(start=27, end=30, label="head", score=1.0),
        )
        assert annotations_from_output[0][1] == (
            "entities",
            LabeledSpan(start=52, end=53, label="head", score=1.0),
        )
        # Assertions for the second document
        assert len(annotations_from_output[1]) == 2
        assert annotations_from_output[1][0] == (
            "entities",
            LabeledSpan(start=27, end=30, label="head", score=1.0),
        )
        assert annotations_from_output[1][1] == (
            "entities",
            LabeledSpan(start=52, end=53, label="head", score=1.0),
        )

    else:
        raise ValueError(f"unknown config: {config}")
