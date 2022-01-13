import os

from datasets import GenerateMode, set_caching_enabled

from pytorch_ie.data.datasets.brat import load_brat, serialize_brat
from tests import FIXTURES_ROOT

TEXT_01 = "Jane lives in Berlin.\n"
TEXT_02 = "Seattle is a rainy city. Jenny Durkan is the city's mayor.\n"
ANNOTS_01 = [
    "T1\tperson 0 4\tJane",
    "T2\tcity 14 20\tBerlin",
]
ANNOTS_02 = [
    "T1\tcity 0 7\tSeattle",
    "T2\tperson 25 37\tJenny Durkan",
    "R1\tmayor_of head:T2 tail:T1",
]


def test_load_brat():
    set_caching_enabled(False)
    dataset = load_brat(
        url=os.path.join(FIXTURES_ROOT, "datasets/brat"),
        conversion_kwargs=dict(head_argument_name="head", tail_argument_name="tail"),
        download_mode=GenerateMode.FORCE_REDOWNLOAD,
    )

    assert isinstance(dataset, dict)
    assert len(dataset) == 1
    assert "train" in dataset
    documents = dataset["train"]
    assert len(documents) == 2

    document = documents[0]
    assert document.text == TEXT_01

    entities = document.annotations("entities")
    assert len(entities) == 2

    entity = entities[0]
    assert entity.start == 0
    assert entity.end == 4
    assert entity.label == "person"
    assert document.text[entity.start : entity.end] == "Jane"

    entity = entities[1]
    assert entity.start == 14
    assert entity.end == 20
    assert entity.label == "city"
    assert document.text[entity.start : entity.end] == "Berlin"

    document = documents[1]
    assert document.text == TEXT_02

    entities = document.annotations("entities")
    assert len(entities) == 2

    entity = entities[0]
    assert entity.start == 0
    assert entity.end == 7
    assert entity.label == "city"
    assert document.text[entity.start : entity.end] == "Seattle"

    entity = entities[1]
    assert entity.start == 25
    assert entity.end == 37
    assert entity.label == "person"
    assert document.text[entity.start : entity.end] == "Jenny Durkan"

    relations = document.annotations("relations")
    assert len(relations) == 1
    relation = relations[0]
    assert relation.label == "mayor_of"
    assert relation.head == entities[1]
    assert relation.tail == entities[0]


def test_serialize_brat_in_memory():
    head_argument_name = "head"
    tail_argument_name = "tail"
    set_caching_enabled(False)
    dataset = load_brat(
        url=os.path.join(FIXTURES_ROOT, "datasets/brat"),
        conversion_kwargs=dict(
            head_argument_name=head_argument_name, tail_argument_name=tail_argument_name
        ),
        download_mode=GenerateMode.FORCE_REDOWNLOAD,
    )

    serialized_brat = serialize_brat(
        dataset, head_argument_name=head_argument_name, tail_argument_name=tail_argument_name
    )
    assert len(serialized_brat) == 1
    assert "train" in serialized_brat
    serialized_docs = list(serialized_brat["train"])
    assert len(serialized_docs) == 2

    serialized_doc = serialized_docs[0]
    assert serialized_doc[0] == "01"
    assert serialized_doc[1] == TEXT_01
    assert serialized_doc[2] == ANNOTS_01

    serialized_doc = serialized_docs[1]
    assert serialized_doc[0] == "02"
    assert serialized_doc[1] == TEXT_02
    assert serialized_doc[2] == ANNOTS_02


def test_serialize_brat_to_directories(tmp_path):
    head_argument_name = "head"
    tail_argument_name = "tail"

    set_caching_enabled(False)
    dataset = load_brat(
        url=os.path.join(FIXTURES_ROOT, "datasets/brat"),
        conversion_kwargs=dict(
            head_argument_name=head_argument_name, tail_argument_name=tail_argument_name
        ),
        download_mode=GenerateMode.FORCE_REDOWNLOAD,
    )

    serialize_brat(
        dataset,
        path=str(tmp_path),
        head_argument_name=head_argument_name,
        tail_argument_name=tail_argument_name,
    )
    with open(tmp_path / "train/01.txt") as f_text:
        text_01 = "".join(f_text.readlines())
    assert text_01 == TEXT_01

    with open(tmp_path / "train/01.ann") as f_text:
        annots_01 = f_text.readlines()
    assert annots_01 == [f"{ann}\n" for ann in ANNOTS_01]

    with open(tmp_path / "train/02.txt") as f_text:
        text_02 = "".join(f_text.readlines())
    assert text_02 == TEXT_02

    with open(tmp_path / "train/02.ann") as f_text:
        annots_02 = f_text.readlines()
    assert annots_02 == [f"{ann}\n" for ann in ANNOTS_02]


def test_full_cycle(tmp_path):
    head_argument_name = "head"
    tail_argument_name = "tail"

    set_caching_enabled(False)
    dataset = load_brat(
        url=os.path.join(FIXTURES_ROOT, "datasets/brat"),
        conversion_kwargs=dict(
            head_argument_name=head_argument_name, tail_argument_name=tail_argument_name
        ),
        download_mode=GenerateMode.FORCE_REDOWNLOAD,
    )
    serialize_brat(
        dataset,
        path=str(tmp_path),
        head_argument_name=head_argument_name,
        tail_argument_name=tail_argument_name,
    )
    set_caching_enabled(False)
    dataset_reloaded = load_brat(
        url=str(tmp_path),
        conversion_kwargs=dict(
            head_argument_name=head_argument_name, tail_argument_name=tail_argument_name
        ),
        download_mode=GenerateMode.FORCE_REDOWNLOAD,
    )

    # assert that datasets (i.e. lists of documents) are equal
    assert dataset.keys() == dataset_reloaded.keys()
    for split in dataset:
        assert len(dataset[split]) == len(
            dataset_reloaded[split]
        ), f"length mismatch for split: {split}"
        for doc, doc_loaded in zip(dataset[split], dataset_reloaded[split]):
            # for now, just compare string representations
            assert str(doc) == str(doc_loaded)
