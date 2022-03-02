import os
from typing import Dict, List

from datasets import GenerateMode, set_caching_enabled

from pytorch_ie import Document
from pytorch_ie.data import BinaryRelation, LabeledSpan
from pytorch_ie.data.datasets.brat import load_brat, serialize_brat, split_span_annotation
from tests import FIXTURES_ROOT
from tests.helpers.document_utils import construct_document

TEXT_01 = "Jane lives in Berlin.\n"
TEXT_02 = "Seattle is a rainy city. Jenny Durkan is the city's mayor.\n"
ANNOTS_01_GENERATED_IDS = ["T3da2b611\tperson 0 4\tJane\n", "T9f2f8b43\tcity 14 20\tBerlin\n"]
ANNOTS_02_GENERATED_IDS = [
    "T3eff2f7a\tcity 0 7\tSeattle\n",
    "T9ddf5dd1\tperson 25 37\tJenny Durkan\n",
    "Rb7123b8e\tmayor_of head:T9ddf5dd1 tail:T3eff2f7a\n",
]
ANNOTS_01_SPECIFIED_IDS = [
    "T1\tperson 0 4\tJane\n",
    "T2\tcity 14 20\tBerlin\n",
]
ANNOTS_02_SPECIFIED_IDS = [
    "T1\tcity 0 7\tSeattle\n",
    "T2\tperson 25 37\tJenny Durkan\n",
    "R1\tmayor_of head:T2 tail:T1\n",
]


def get_doc1(with_ids: bool = False, **kwargs) -> Document:
    ent1 = LabeledSpan(start=0, end=4, label="person", metadata={"text": "Jane"})
    ent2 = LabeledSpan(start=14, end=20, label="city", metadata={"text": "Berlin"})
    doc = construct_document(text=TEXT_01, entities=[ent1, ent2], relations=[], **kwargs)

    if with_ids:
        ent1.metadata["id"] = "1"
        ent2.metadata["id"] = "2"
    return doc


def get_doc2(with_ids: bool = False, **kwargs) -> Document:
    text = TEXT_02
    ent1 = LabeledSpan(start=0, end=7, label="city", metadata={"text": "Seattle"})
    ent2 = LabeledSpan(start=25, end=37, label="person", metadata={"text": "Jenny Durkan"})
    rel1 = BinaryRelation(head=ent2, tail=ent1, label="mayor_of")
    doc = construct_document(text=text, entities=[ent1, ent2], relations=[rel1], **kwargs)
    if with_ids:
        ent1.metadata["id"] = "1"
        ent2.metadata["id"] = "2"
        rel1.metadata["id"] = "1"
    return doc


def get_dataset(split_name: str = "train", with_ids: bool = False) -> Dict[str, List[Document]]:
    entity_annotation_name = "entities"
    relation_annotation_name = "relations"
    dataset = {
        split_name: [
            get_doc1(
                entity_annotation_name=entity_annotation_name,
                relation_annotation_name=relation_annotation_name,
                with_ids=with_ids,
            ),
            get_doc2(
                entity_annotation_name=entity_annotation_name,
                relation_annotation_name=relation_annotation_name,
                with_ids=with_ids,
            ),
        ]
    }
    return dataset


def assert_dataset_equal(dataset, other):
    # assert that datasets (i.e. lists of documents) are equal
    assert dataset.keys() == other.keys()
    for split in dataset:
        assert len(dataset[split]) == len(other[split]), f"length mismatch for split: {split}"
        for doc, doc_loaded in zip(dataset[split], other[split]):
            # for now, just compare string representations
            assert str(doc) == str(doc_loaded)


def test_load_brat():
    dataset = get_dataset(with_ids=True)
    set_caching_enabled(False)
    dataset_loaded = load_brat(
        url=os.path.join(FIXTURES_ROOT, "datasets/brat"),
        conversion_kwargs=dict(head_argument_name="head", tail_argument_name="tail"),
        download_mode=GenerateMode.FORCE_REDOWNLOAD,
    )
    assert_dataset_equal(dataset, dataset_loaded)


def test_load_and_serialize_brat_in_memory():
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
    assert serialized_doc[0] == "1"
    assert serialized_doc[1] == TEXT_01
    assert serialized_doc[2] == ANNOTS_01_SPECIFIED_IDS

    serialized_doc = serialized_docs[1]
    assert serialized_doc[0] == "2"
    assert serialized_doc[1] == TEXT_02
    assert serialized_doc[2] == ANNOTS_02_SPECIFIED_IDS


def test_serialize_brat_with_construct_ids_in_memory():
    head_argument_name = "head"
    tail_argument_name = "tail"

    dataset = get_dataset(with_ids=False)
    serialized_brat = serialize_brat(
        dataset,
        head_argument_name=head_argument_name,
        tail_argument_name=tail_argument_name,
    )
    assert len(serialized_brat) == 1
    assert "train" in serialized_brat
    serialized_docs = list(serialized_brat["train"])
    assert len(serialized_docs) == 2

    serialized_doc = serialized_docs[0]
    assert serialized_doc[0] is None
    assert serialized_doc[1] == TEXT_01
    assert serialized_doc[2] == ANNOTS_01_GENERATED_IDS

    serialized_doc = serialized_docs[1]
    assert serialized_doc[0] is None
    assert serialized_doc[1] == TEXT_02
    assert serialized_doc[2] == ANNOTS_02_GENERATED_IDS


def test_serialize_brat_to_directories(tmp_path):

    dataset = get_dataset(with_ids=False)

    head_argument_name = "head"
    tail_argument_name = "tail"
    serialize_brat(
        dataset,
        path=str(tmp_path),
        head_argument_name=head_argument_name,
        tail_argument_name=tail_argument_name,
    )
    doc1_id = "c56d9dd8"
    with open(tmp_path / f"train/{doc1_id}.txt") as f_text:
        text_01 = "".join(f_text.readlines())
    assert text_01 == TEXT_01

    with open(tmp_path / f"train/{doc1_id}.ann") as f_text:
        annots_01 = f_text.readlines()
    assert annots_01 == ANNOTS_01_GENERATED_IDS

    doc2_id = "a92df0b9"
    with open(tmp_path / f"train/{doc2_id}.txt") as f_text:
        text_02 = "".join(f_text.readlines())
    assert text_02 == TEXT_02

    with open(tmp_path / f"train/{doc2_id}.ann") as f_text:
        annots_02 = f_text.readlines()
    assert annots_02 == ANNOTS_02_GENERATED_IDS


def test_load_and_serialize_and_load_brat(tmp_path):
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
    dataset_loaded = load_brat(
        url=str(tmp_path),
        conversion_kwargs=dict(
            head_argument_name=head_argument_name, tail_argument_name=tail_argument_name
        ),
        download_mode=GenerateMode.FORCE_REDOWNLOAD,
    )

    assert_dataset_equal(dataset, dataset_loaded)


def test_split_span_annotation():

    text_wo_nl = "This is a text without newlines."
    span_slice = (0, len(text_wo_nl))
    slices = split_span_annotation(text=text_wo_nl, slice=span_slice, glue="\n")
    assert slices == [span_slice]

    span_slice = (3, 15)
    slices = split_span_annotation(text=text_wo_nl, slice=span_slice, glue="\n")
    assert slices == [span_slice]

    text_with_nl = "This is a text\nwith\nnewlines."
    span_slice = (0, len(text_with_nl))
    slices = split_span_annotation(text=text_with_nl, slice=span_slice, glue="\n")
    assert slices == [(0, 14), (15, 19), (20, 29)]

    span_slice = (3, 18)
    slices = split_span_annotation(text=text_with_nl, slice=span_slice, glue="\n")
    assert slices == [(3, 14), (15, 18)]
