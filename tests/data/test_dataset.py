import datasets
import pytest

from tests import FIXTURES_ROOT
from tests.data.dataset_tester import DatasetTester


def test_dataset(document_dataset):
    assert set(document_dataset.keys()) == {"train", "validation", "test"}

    assert len(document_dataset["train"]) == 8
    assert len(document_dataset["validation"]) == 2
    assert len(document_dataset["test"]) == 2

    train_doc5 = document_dataset["train"][4]
    assert train_doc5.id == "train_doc5"
    assert len(train_doc5.sentences) == 3
    assert len(train_doc5.entities) == 3
    assert len(train_doc5.relations) == 3

    assert train_doc5.sentences[1].text == "Entity G works at H."


def test_dataset_index(document_dataset):
    train_dataset = document_dataset["train"]
    assert train_dataset[4].id == "train_doc5"
    assert [doc.id for doc in train_dataset[0, 3, 5]] == ["train_doc1", "train_doc4", "train_doc6"]
    assert [doc.id for doc in train_dataset[2:5]] == ["train_doc3", "train_doc4", "train_doc5"]


def test_dataset_map(document_dataset):
    train_dataset = document_dataset["train"]

    def clear_relations(document):
        document.relations.clear()
        return document

    assert sum(len(doc.relations) for doc in train_dataset) == 7

    mapped_dataset1 = train_dataset.map(clear_relations)

    assert sum(len(doc.relations) for doc in mapped_dataset1) == 0
    assert sum(len(doc.relations) for doc in train_dataset) == 7


def test_dataset_map_batched(document_dataset):
    train_dataset = document_dataset["train"]

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
