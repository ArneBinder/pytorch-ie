import copy
import os
from typing import Optional

import pytest
import torch

from pytorch_ie import Document
from pytorch_ie.data import BinaryRelation, LabeledSpan
from pytorch_ie.taskmodules import TransformerRETextClassificationTaskModule
from pytorch_ie.taskmodules.transformer_re_text_classification import (
    _enumerate_entity_pairs,
    _get_window_around_slice,
)

TEXT_01 = "Jane lives in Berlin. this is no sentence about Karl\n"
TEXT_02 = "Seattle is a rainy city. Jenny Durkan is the city's mayor.\n"
TEXT_03 = "Karl enjoys sunny days in Berlin."


def _add_entity(
    doc: Document,
    start: int,
    end: int,
    label: str,
    annotation_name: str,
    assert_text: str,
    id: Optional[str] = None,
) -> LabeledSpan:
    ent = LabeledSpan(start=start, end=end, label=label)
    ent.metadata["text"] = doc.text[ent.start : ent.end]
    assert ent.metadata["text"] == assert_text
    if id is not None:
        ent.metadata["id"] = id
    doc.add_annotation(name=annotation_name, annotation=ent)
    return ent


def _add_relation(
    doc: Document,
    head: LabeledSpan,
    tail: LabeledSpan,
    label: str,
    annotation_name: str,
    id: Optional[str] = None,
) -> BinaryRelation:
    rel = BinaryRelation(head=head, tail=tail, label=label)
    if id is not None:
        rel.metadata["id"] = id
    doc.add_annotation(name=annotation_name, annotation=rel)
    return rel


def get_doc1(
    entity_annotation_name: str,
    relation_annotation_name: str,
    sentence_annotation_name: str,
    with_ids: bool = False,
    **kwargs,
) -> Document:
    doc = Document(text=TEXT_01, doc_id="1" if with_ids else None)
    ent1 = _add_entity(
        doc=doc,
        start=0,
        end=4,
        label="person",
        assert_text="Jane",
        annotation_name=entity_annotation_name,
        id="1" if with_ids else None,
    )
    ent2 = _add_entity(
        doc=doc,
        start=14,
        end=20,
        label="city",
        assert_text="Berlin",
        annotation_name=entity_annotation_name,
        id="2" if with_ids else None,
    )
    _add_entity(
        doc=doc,
        start=48,
        end=52,
        label="person",
        assert_text="Karl",
        annotation_name=entity_annotation_name,
        id="3" if with_ids else None,
    )
    _add_entity(
        doc=doc,
        start=0,
        end=21,
        label="sentence",
        assert_text="Jane lives in Berlin.",
        annotation_name=sentence_annotation_name,
        id="4" if with_ids else None,
    )
    _add_relation(
        doc=doc,
        head=ent1,
        tail=ent2,
        label="lives_in",
        annotation_name=relation_annotation_name,
        id="1" if with_ids else None,
    )
    return doc


def get_doc2(
    entity_annotation_name: str,
    relation_annotation_name: str,
    sentence_annotation_name: str,
    with_ids: bool = False,
) -> Document:
    doc = Document(text=TEXT_02, doc_id="3" if with_ids else None)
    ent1 = _add_entity(
        doc=doc,
        start=0,
        end=7,
        label="city",
        assert_text="Seattle",
        annotation_name=entity_annotation_name,
        id="1" if with_ids else None,
    )
    ent2 = _add_entity(
        doc=doc,
        start=25,
        end=37,
        label="person",
        assert_text="Jenny Durkan",
        annotation_name=entity_annotation_name,
        id="2" if with_ids else None,
    )
    _add_entity(
        doc=doc,
        start=0,
        end=24,
        label="sentence",
        assert_text="Seattle is a rainy city.",
        annotation_name=sentence_annotation_name,
        id="3" if with_ids else None,
    )
    _add_entity(
        doc=doc,
        start=25,
        end=58,
        label="sentence",
        assert_text="Jenny Durkan is the city's mayor.",
        annotation_name=sentence_annotation_name,
        id="4" if with_ids else None,
    )
    _add_relation(
        doc=doc,
        head=ent2,
        tail=ent1,
        label="mayor_of",
        annotation_name=relation_annotation_name,
        id="1" if with_ids else None,
    )
    return doc


def get_doc3(
    entity_annotation_name: str,
    relation_annotation_name: str,
    sentence_annotation_name: str,
    with_ids: bool = False,
) -> Document:
    doc = Document(text=TEXT_03, doc_id="2" if with_ids else None)
    _add_entity(
        doc=doc,
        start=0,
        end=4,
        label="person",
        assert_text="Karl",
        annotation_name=entity_annotation_name,
        id="1" if with_ids else None,
    )
    _add_entity(
        doc=doc,
        start=26,
        end=32,
        label="city",
        assert_text="Berlin",
        annotation_name=entity_annotation_name,
        id="2" if with_ids else None,
    )
    _add_entity(
        doc=doc,
        start=0,
        end=33,
        label="sentence",
        assert_text="Karl enjoys sunny days in Berlin.",
        annotation_name=sentence_annotation_name,
        id="3" if with_ids else None,
    )
    # TODO: this is kind of hacky
    doc._annotations[relation_annotation_name] = []
    return doc


@pytest.fixture
def documents():
    doc_kwargs = dict(
        entity_annotation_name="entities",
        relation_annotation_name="relations",
        sentence_annotation_name="sentences",
    )
    # TODO: add doc3: this should not change anything (no new relation candidates), but it does!
    # maybe implement test_enumerate_entity_pairs() before checking the encode methods
    documents = sorted(
        [get_doc1(**doc_kwargs), get_doc2(**doc_kwargs)],  # get_doc3(**doc_kwargs)],
        key=lambda doc: doc.text,
    )
    return documents


@pytest.fixture(scope="module")
def taskmodule():
    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = TransformerRETextClassificationTaskModule(
        tokenizer_name_or_path=tokenizer_name_or_path,
    )
    return taskmodule


@pytest.fixture
def prepared_taskmodule(taskmodule, documents):
    taskmodule.prepare(documents)
    return taskmodule


@pytest.fixture(scope="module", params=[False, True])
def taskmodule_optional_marker(request):
    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = TransformerRETextClassificationTaskModule(
        tokenizer_name_or_path=tokenizer_name_or_path, add_type_to_marker=request.param
    )
    return taskmodule


@pytest.fixture
def prepared_taskmodule_optional_marker(taskmodule_optional_marker, documents):
    taskmodule_optional_marker.prepare(documents)
    return taskmodule_optional_marker


@pytest.fixture
def model_output():
    return {
        "logits": torch.tensor(
            [
                [-1.6924, 9.5473, -1.9625],
                [-0.9995, -2.5705, 10.0095],
            ]
        ),
    }


def test_prepare(taskmodule_optional_marker, documents):
    assert not taskmodule_optional_marker.is_prepared()
    taskmodule_optional_marker.prepare(documents)
    assert taskmodule_optional_marker.is_prepared()
    assert set(taskmodule_optional_marker.label_to_id.keys()) == {
        "no_relation",
        "mayor_of",
        "lives_in",
    }
    assert taskmodule_optional_marker.label_to_id["no_relation"] == 0
    if taskmodule_optional_marker.add_type_to_marker:
        assert taskmodule_optional_marker.argument_markers == {
            ("head", "end", "city"): "[/H:city]",
            ("head", "end", "person"): "[/H:person]",
            ("tail", "end", "city"): "[/T:city]",
            ("tail", "end", "person"): "[/T:person]",
            ("head", "start", "city"): "[H:city]",
            ("head", "start", "person"): "[H:person]",
            ("tail", "start", "city"): "[T:city]",
            ("tail", "start", "person"): "[T:person]",
        }
    else:
        assert taskmodule_optional_marker.argument_markers == {
            ("head", "end"): "[/H]",
            ("tail", "end"): "[/T]",
            ("head", "start"): "[H]",
            ("tail", "start"): "[T]",
        }


def test_config(prepared_taskmodule_optional_marker):
    config = prepared_taskmodule_optional_marker._config()
    assert config["taskmodule_type"] == "TransformerRETextClassificationTaskModule"
    assert "label_to_id" in config
    assert set(config["label_to_id"]) == {"no_relation", "mayor_of", "lives_in"}
    if prepared_taskmodule_optional_marker.add_type_to_marker:
        assert set(config["entity_labels"]) == {"person", "city"}
    else:
        assert config["entity_labels"] == []


def test_encode_input(prepared_taskmodule_optional_marker, documents):
    (
        input_encoding,
        metadata,
        new_documents,
    ) = prepared_taskmodule_optional_marker.encode_input(documents)
    assert len(input_encoding) == 2
    assert new_documents is not None
    assert len(new_documents) == 2
    encoding = input_encoding[0]
    document = new_documents[0]
    assert document.text == TEXT_01
    if prepared_taskmodule_optional_marker.add_type_to_marker:
        assert prepared_taskmodule_optional_marker.tokenizer.convert_ids_to_tokens(
            encoding["input_ids"]
        ) == [
            "[CLS]",
            "[H:person]",
            "Jane",
            "[/H:person]",
            "lives",
            "in",
            "[T:city]",
            "Berlin",
            "[/T:city]",
            ".",
            "this",
            "is",
            "no",
            "sentence",
            "about",
            "Karl",
            "[SEP]",
        ]
    else:
        assert prepared_taskmodule_optional_marker.tokenizer.convert_ids_to_tokens(
            encoding["input_ids"]
        ) == [
            "[CLS]",
            "[H]",
            "Jane",
            "[/H]",
            "lives",
            "in",
            "[T]",
            "Berlin",
            "[/T]",
            ".",
            "this",
            "is",
            "no",
            "sentence",
            "about",
            "Karl",
            "[SEP]",
        ]

    encoding = input_encoding[1]
    document = new_documents[1]
    assert document.text == TEXT_02
    if prepared_taskmodule_optional_marker.add_type_to_marker:
        assert prepared_taskmodule_optional_marker.tokenizer.convert_ids_to_tokens(
            encoding["input_ids"]
        ) == [
            "[CLS]",
            "[T:city]",
            "Seattle",
            "[/T:city]",
            "is",
            "a",
            "rainy",
            "city",
            ".",
            "[H:person]",
            "Jenny",
            "Du",
            "##rka",
            "##n",
            "[/H:person]",
            "is",
            "the",
            "city",
            "'",
            "s",
            "mayor",
            ".",
            "[SEP]",
        ]
    else:
        assert prepared_taskmodule_optional_marker.tokenizer.convert_ids_to_tokens(
            encoding["input_ids"]
        ) == [
            "[CLS]",
            "[T]",
            "Seattle",
            "[/T]",
            "is",
            "a",
            "rainy",
            "city",
            ".",
            "[H]",
            "Jenny",
            "Du",
            "##rka",
            "##n",
            "[/H]",
            "is",
            "the",
            "city",
            "'",
            "s",
            "mayor",
            ".",
            "[SEP]",
        ]


@pytest.mark.parametrize("encode_target", [False, True])
def test_encode_target(prepared_taskmodule, documents, encode_target):
    task_encodings = prepared_taskmodule.encode(documents, encode_target=encode_target)
    assert len(task_encodings) == 2
    if encode_target:
        encoding = task_encodings[0]
        assert encoding.has_target
        target_labels = [prepared_taskmodule.id_to_label[_id] for _id in encoding.target]
        assert target_labels == ["lives_in"]

        encoding = task_encodings[1]
        assert encoding.has_target
        target_labels = [prepared_taskmodule.id_to_label[_id] for _id in encoding.target]
        assert target_labels == ["mayor_of"]
    else:
        assert [encoding.has_target for encoding in task_encodings] == [False, False]


@pytest.mark.parametrize("encode_target", [False, True])
def test_encode(prepared_taskmodule_optional_marker, documents, encode_target):
    # the code is actually tested in test_encode_input() and test_encode_target(). Here we only test assertions in encode().
    task_encodings = prepared_taskmodule_optional_marker.encode(documents, encode_target=True)


def test_encode_input_with_partitions(prepared_taskmodule_optional_marker, documents):
    prepared_taskmodule_with_partitions = copy.deepcopy(prepared_taskmodule_optional_marker)
    prepared_taskmodule_with_partitions.partition_annotation = "sentences"
    input_encoding, metadata, new_documents = prepared_taskmodule_with_partitions.encode_input(
        documents
    )
    assert len(input_encoding) == 1
    assert new_documents is not None
    assert len(new_documents) == 1
    encoding = input_encoding[0]
    if prepared_taskmodule_with_partitions.add_type_to_marker:
        assert prepared_taskmodule_optional_marker.tokenizer.convert_ids_to_tokens(
            encoding["input_ids"]
        ) == [
            "[CLS]",
            "[H:person]",
            "Jane",
            "[/H:person]",
            "lives",
            "in",
            "[T:city]",
            "Berlin",
            "[/T:city]",
            ".",
            "[SEP]",
        ]
    else:
        assert prepared_taskmodule_optional_marker.tokenizer.convert_ids_to_tokens(
            encoding["input_ids"]
        ) == [
            "[CLS]",
            "[H]",
            "Jane",
            "[/H]",
            "lives",
            "in",
            "[T]",
            "Berlin",
            "[/T]",
            ".",
            "[SEP]",
        ]


def test_encode_with_windowing(prepared_taskmodule_optional_marker, documents):
    prepared_taskmodule_with_windowing = copy.deepcopy(prepared_taskmodule_optional_marker)
    prepared_taskmodule_with_windowing.max_window = 10
    task_encodings = prepared_taskmodule_with_windowing.encode(documents, encode_target=False)
    assert len(task_encodings) == 1

    encoding = task_encodings[0]
    document = documents[0]
    assert encoding.document == document
    assert "input_ids" in encoding.input
    assert len(encoding.input["input_ids"]) <= prepared_taskmodule_with_windowing.max_window
    if prepared_taskmodule_with_windowing.add_type_to_marker:
        assert prepared_taskmodule_with_windowing.tokenizer.convert_ids_to_tokens(
            encoding.input["input_ids"]
        ) == [
            "[CLS]",
            "[H:person]",
            "Jane",
            "[/H:person]",
            "lives",
            "in",
            "[T:city]",
            "Berlin",
            "[/T:city]",
            "[SEP]",
        ]
    else:
        assert prepared_taskmodule_with_windowing.tokenizer.convert_ids_to_tokens(
            encoding.input["input_ids"]
        ) == ["[CLS]", "[H]", "Jane", "[/H]", "lives", "in", "[T]", "Berlin", "[/T]", "[SEP]"]


@pytest.mark.parametrize("encode_target", [False, True])
def test_collate(prepared_taskmodule_optional_marker, documents, encode_target):
    encodings = prepared_taskmodule_optional_marker.encode(documents, encode_target=encode_target)
    assert len(encodings) == 2

    if encode_target:
        assert all([encoding.has_target for encoding in encodings])
    else:
        assert not any([encoding.has_target for encoding in encodings])

    batch_encoding = prepared_taskmodule_optional_marker.collate(encodings)
    inputs, targets = batch_encoding
    assert "input_ids" in inputs
    assert "attention_mask" in inputs
    assert inputs["input_ids"].shape[0] == 2
    assert inputs["input_ids"].shape == inputs["attention_mask"].shape

    if prepared_taskmodule_optional_marker.add_type_to_marker:
        tokens1 = prepared_taskmodule_optional_marker.tokenizer.convert_ids_to_tokens(
            inputs["input_ids"].tolist()[0]
        )
        assert tokens1 == [
            "[CLS]",
            "[H:person]",
            "Jane",
            "[/H:person]",
            "lives",
            "in",
            "[T:city]",
            "Berlin",
            "[/T:city]",
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
        tokens2 = prepared_taskmodule_optional_marker.tokenizer.convert_ids_to_tokens(
            inputs["input_ids"].tolist()[1]
        )
        assert tokens2 == [
            "[CLS]",
            "[T:city]",
            "Seattle",
            "[/T:city]",
            "is",
            "a",
            "rainy",
            "city",
            ".",
            "[H:person]",
            "Jenny",
            "Du",
            "##rka",
            "##n",
            "[/H:person]",
            "is",
            "the",
            "city",
            "'",
            "s",
            "mayor",
            ".",
            "[SEP]",
        ]

    else:
        tokens1 = prepared_taskmodule_optional_marker.tokenizer.convert_ids_to_tokens(
            inputs["input_ids"].tolist()[0]
        )
        assert tokens1 == [
            "[CLS]",
            "[H]",
            "Jane",
            "[/H]",
            "lives",
            "in",
            "[T]",
            "Berlin",
            "[/T]",
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
        tokens2 = prepared_taskmodule_optional_marker.tokenizer.convert_ids_to_tokens(
            inputs["input_ids"].tolist()[1]
        )
        assert tokens2 == [
            "[CLS]",
            "[T]",
            "Seattle",
            "[/T]",
            "is",
            "a",
            "rainy",
            "city",
            ".",
            "[H]",
            "Jenny",
            "Du",
            "##rka",
            "##n",
            "[/H]",
            "is",
            "the",
            "city",
            "'",
            "s",
            "mayor",
            ".",
            "[SEP]",
        ]

    if encode_target:
        assert targets.shape == (2,)
        labels = [
            prepared_taskmodule_optional_marker.id_to_label[target_id]
            for target_id in targets.tolist()
        ]
        assert labels == ["lives_in", "mayor_of"]
    else:
        assert targets is None


def test_unbatch_output(prepared_taskmodule, model_output):
    unbatched_outputs = prepared_taskmodule.unbatch_output(model_output)

    assert len(unbatched_outputs) == 2

    unbatched_output1 = unbatched_outputs[0]
    assert len(unbatched_output1["labels"]) == 1
    assert len(unbatched_output1["probabilities"]) == 1
    assert unbatched_output1["labels"][0] == "lives_in"
    assert unbatched_output1["probabilities"][0] == pytest.approx(0.9999768733978271)

    unbatched_output2 = unbatched_outputs[1]
    assert len(unbatched_output2["labels"]) == 1
    assert len(unbatched_output2["probabilities"]) == 1
    assert unbatched_output2["labels"][0] == "mayor_of"
    assert unbatched_output2["probabilities"][0] == pytest.approx(0.9999799728393555)


@pytest.mark.parametrize("inplace", [True, False])
def test_decode(prepared_taskmodule, documents, model_output, inplace):
    encodings = prepared_taskmodule.encode(documents, encode_target=False)
    unbatched_outputs = prepared_taskmodule.unbatch_output(model_output)
    decoded_documents = prepared_taskmodule.decode(
        encodings=encodings, decoded_outputs=unbatched_outputs, inplace=inplace
    )

    assert len(decoded_documents) == len(documents)
    if inplace:
        assert set(decoded_documents) == set(documents)
    else:
        assert set(decoded_documents).isdisjoint(set(documents))

    # sort documents because order of documents is not deterministic if inplace==False
    decoded_documents = sorted(decoded_documents, key=lambda doc: doc.text)
    assert len(decoded_documents) == 2

    predictions = decoded_documents[0].predictions("relations")
    assert len(predictions) == 1
    prediction = predictions[0]
    assert prediction.label == "lives_in"
    head = prediction.head
    assert head.label == "person"
    assert head.start == 0
    assert head.end == 4
    tail = prediction.tail
    assert tail.label == "city"
    assert tail.start == 14
    assert tail.end == 20

    predictions = decoded_documents[1].predictions("relations")
    assert len(predictions) == 1
    prediction = predictions[0]
    assert prediction.label == "mayor_of"
    head = prediction.head
    assert head.label == "person"
    assert head.start == 25
    assert head.end == 37
    tail = prediction.tail
    assert tail.label == "city"
    assert tail.start == 0
    assert tail.end == 7


def test_save_load(tmp_path, prepared_taskmodule):
    path = os.path.join(tmp_path, "taskmodule")
    prepared_taskmodule.save_pretrained(path)
    loaded_taskmodule = TransformerRETextClassificationTaskModule.from_pretrained(path)
    assert loaded_taskmodule.is_prepared()
    assert loaded_taskmodule.argument_markers == prepared_taskmodule.argument_markers


def test_get_window_around_slice():

    # default: result is centered around slice
    window_slice = _get_window_around_slice(
        slice=(5, 7), max_window_size=6, available_input_length=10
    )
    assert window_slice == (3, 9)

    # slice at the beginning -> shift window to the right (regarding the slice center)
    window_slice = _get_window_around_slice(
        slice=(0, 5), max_window_size=8, available_input_length=10
    )
    assert window_slice == (0, 8)

    # slice at the end -> shift window to the left (regarding the slice center)
    window_slice = _get_window_around_slice(
        slice=(7, 10), max_window_size=8, available_input_length=10
    )
    assert window_slice == (2, 10)

    # max window size bigger than available_input_length -> take everything
    window_slice = _get_window_around_slice(
        slice=(2, 6), max_window_size=8, available_input_length=7
    )
    assert window_slice == (0, 7)

    # slice exceeds max_window_size
    window_slice = _get_window_around_slice(
        slice=(0, 5), max_window_size=4, available_input_length=10
    )
    assert window_slice is None


def test_enumerate_entity_pairs():
    # TODO
    # Especially check, what's happening when _no_ relations are given (maybe also create another test for encode_input)
    # The current code assumes that relation annotations are available for all documents (at least an empty list) or
    # not (None).
    pass

    head, head_token_slice, tail, tail_token_slice = enumerated_entity_pairs[0]
