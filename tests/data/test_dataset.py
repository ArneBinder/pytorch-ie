import re

import datasets
import numpy
import pytest
import torch

from pytorch_ie.taskmodules import (
    InplaceNotSupportedException,
    TransformerSpanClassificationTaskModule,
)
from tests import FIXTURES_ROOT
from tests.data.dataset_tester import DatasetTester


@pytest.fixture(scope="module")
def taskmodule():
    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = TransformerSpanClassificationTaskModule(
        tokenizer_name_or_path=tokenizer_name_or_path
    )
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


def test_dataset(dataset):
    assert set(dataset.keys()) == {"train", "validation", "test"}

    assert len(dataset["train"]) == 8
    assert len(dataset["validation"]) == 2
    assert len(dataset["test"]) == 2

    train_doc5 = dataset["train"][4]
    assert train_doc5.id == "train_doc5"
    assert len(train_doc5.sentences) == 3
    assert len(train_doc5.entities) == 3
    assert len(train_doc5.relations) == 3

    assert str(train_doc5.sentences[1]) == "Entity G works at H."


def test_dataset_index(dataset):
    train_dataset = dataset["train"]
    assert train_dataset[4].id == "train_doc5"
    assert [doc.id for doc in train_dataset[0, 3, 5]] == ["train_doc1", "train_doc4", "train_doc6"]
    assert [doc.id for doc in train_dataset[2:5]] == ["train_doc3", "train_doc4", "train_doc5"]


def test_dataset_map(dataset):
    train_dataset = dataset["train"]

    def clear_relations(document):
        document.relations.clear()
        return document

    assert sum(len(doc.relations) for doc in train_dataset) == 7

    mapped_dataset1 = train_dataset.map(clear_relations)

    assert sum(len(doc.relations) for doc in mapped_dataset1) == 0
    assert sum(len(doc.relations) for doc in train_dataset) == 7


def test_dataset_map_batched(dataset):
    train_dataset = dataset["train"]

    def clear_relations_batched(documents):
        assert len(documents) == 2
        for document in documents:
            document.relations.clear()
        return documents

    assert sum(len(doc.relations) for doc in train_dataset) == 7

    mapped_dataset1 = train_dataset.map(clear_relations_batched, batched=True, batch_size=2)

    assert sum(len(doc.relations) for doc in mapped_dataset1) == 0
    assert sum(len(doc.relations) for doc in train_dataset) == 7


def test_with_tester():
    dataset_name = FIXTURES_ROOT / "datasets" / "conll2003"

    dataset_tester = DatasetTester(parent=None)
    configs = dataset_tester.load_all_configs(dataset_name, is_local=True)[:1]
    dataset_tester.check_load_dataset(
        dataset_name=dataset_name, configs=configs, is_local=True, use_local_dummy_data=True
    )


@pytest.mark.parametrize("encode_target", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
def test_dataset_with_taskmodule(dataset, taskmodule, model_output, encode_target, inplace):
    train_dataset = dataset["train"]

    taskmodule.prepare(train_dataset)
    assert set(taskmodule.label_to_id.keys()) == {"PER", "ORG", "O"}
    assert [taskmodule.id_to_label[i] for i in range(3)] == ["O", "ORG", "PER"]
    assert taskmodule.label_to_id["O"] == 0

    task_encodings = taskmodule.encode(train_dataset, encode_target=encode_target)
    assert len(task_encodings) == 8

    task_encoding = task_encodings[5]
    document = train_dataset[5]
    assert task_encoding.document == document
    assert "input_ids" in task_encoding.inputs
    assert (
        taskmodule.tokenizer.decode(task_encoding.inputs["input_ids"], skip_special_tokens=True)
        == document.text
    )

    if encode_target:
        assert task_encoding.targets == [
            (1, 4, taskmodule.label_to_id["PER"]),
            (6, 6, taskmodule.label_to_id["ORG"]),
            (9, 9, taskmodule.label_to_id["ORG"]),
        ]
    else:
        assert not task_encoding.has_targets

    unbatched_outputs = taskmodule.unbatch_output(model_output)

    decoded_documents = taskmodule.decode(
        task_encodings=task_encodings,
        task_outputs=unbatched_outputs,
        inplace=inplace,
    )

    assert len(decoded_documents) == len(train_dataset)

    assert {id(doc) for doc in decoded_documents}.isdisjoint({id(doc) for doc in train_dataset})

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

    for document in train_dataset:
        assert not document["entities"].predictions


@pytest.mark.slow
def test_load_with_hf_datasets():
    dataset_dir = FIXTURES_ROOT / "datasets" / "conll2003"

    dataset = datasets.load_dataset(
        path=str(dataset_dir / "conll2003.py"),
    )

    assert set(dataset.keys()) == {"train", "validation", "test"}

    # TODO: the updated CoNLL03 data files have two newlines at the end
    # this results in one additional example in train, validation, and test
    # --> file a bug report in HF datasets
    assert len(dataset["train"]) == 14042  # 14041
    assert len(dataset["validation"]) == 3251
    assert len(dataset["test"]) == 3454


@pytest.mark.slow
def test_load_with_hf_datasets_from_hub():
    dataset = datasets.load_dataset(
        path="pie/conll2003",
    )

    assert set(dataset.keys()) == {"train", "validation", "test"}

    # TODO: the updated CoNLL03 data files have two newlines at the end
    # this results in one additional example in train, validation, and test
    # --> file a bug report in HF datasets
    assert len(dataset["train"]) == 14042  # 14041
    assert len(dataset["validation"]) == 3251
    assert len(dataset["test"]) == 3454
