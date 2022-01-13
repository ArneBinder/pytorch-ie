import os

from datasets import GenerateMode, set_caching_enabled

from pytorch_ie.data.datasets.brat import load_brat, serialize_brat
from tests import FIXTURES_ROOT


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
    assert document.text == "Jane lives in Berlin.\n"

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
    assert document.text == "Seattle is a rainy city. Jenny Durkan is the city's mayor.\n"

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


def test_serialize_brat():
    set_caching_enabled(False)
    dataset = load_brat(
        url=os.path.join(FIXTURES_ROOT, "datasets/brat"),
        conversion_kwargs=dict(head_argument_name="head", tail_argument_name="tail"),
        download_mode=GenerateMode.FORCE_REDOWNLOAD,
    )

    serialized_brat = serialize_brat(dataset)
    assert len(serialized_brat) == 1
    assert "train" in serialized_brat
    serialized_docs = list(serialized_brat["train"])
    assert len(serialized_docs) == 2

    serialized_doc = serialized_docs[0]
    assert serialized_doc[0] == "01"
    assert serialized_doc[1] == "Jane lives in Berlin.\n"
    assert serialized_doc[2] == ['T1\tperson 0 4\tJane', 'T2\tcity 14 20\tBerlin']

    serialized_doc = serialized_docs[1]
    assert serialized_doc[0] == "02"
    assert serialized_doc[1] == "Seattle is a rainy city. Jenny Durkan is the city's mayor.\n"
    assert serialized_doc[2] == ['T1\tcity 0 7\tSeattle', 'T2\tperson 25 37\tJenny Durkan', 'R1\tmayor_of Arg1:T2 Arg2:T1']

    # TODO: write into temp folder and check that content
    #serialized_brat = serialize_brat(dataset, path=...)
