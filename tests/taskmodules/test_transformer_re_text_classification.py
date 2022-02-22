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

DOC1_ENTITY_JANE = LabeledSpan(start=0, end=4, label="person", metadata={"text": "Jane"})
DOC1_ENTITY_BERLIN = LabeledSpan(start=14, end=20, label="city", metadata={"text": "Berlin"})
DOC1_ENTITY_KARL = LabeledSpan(start=48, end=52, label="person", metadata={"text": "Karl"})
DOC1_SENTENCE1 = LabeledSpan(
    start=0, end=21, label="sentence", metadata={"text": "Jane lives in Berlin."}
)
DOC1_REL_LIVES_IN = BinaryRelation(
    head=DOC1_ENTITY_JANE, tail=DOC1_ENTITY_BERLIN, label="lives_in"
)

DOC2_ENTITY_SEATTLE = LabeledSpan(start=0, end=7, label="city", metadata={"text": "Seattle"})
DOC2_ENTITY_JENNY = LabeledSpan(
    start=25, end=37, label="person", metadata={"text": "Jenny Durkan"}
)
DOC2_SENTENCE1 = LabeledSpan(
    start=0, end=24, label="sentence", metadata={"text": "Seattle is a rainy city."}
)
DOC2_SENTENCE2 = LabeledSpan(
    start=25, end=58, label="sentence", metadata={"text": "Jenny Durkan is the city's mayor."}
)
DOC2_REL_MAYOR_OF = BinaryRelation(
    head=DOC2_ENTITY_JENNY, tail=DOC2_ENTITY_SEATTLE, label="mayor_of"
)

DOC3_ENTITY_KARL = LabeledSpan(start=0, end=4, label="person", metadata={"text": "Karl"})
DOC3_ENTITY_BERLIN = LabeledSpan(start=26, end=32, label="city", metadata={"text": "Berlin"})
DOC3_SENTENCE1 = LabeledSpan(
    start=0, end=33, label="sentence", metadata={"text": "Karl enjoys sunny days in Berlin."}
)

DOC1_TOKENS = [
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
DOC2_TOKENS = [
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
DOC3_TOKENS = ["[CLS]", "Karl", "enjoys", "sunny", "days", "in", "Berlin", ".", "[SEP]"]


def _add_span(
    doc: Document,
    span: LabeledSpan,
    annotation_name: str,
    id: Optional[str] = None,
) -> LabeledSpan:
    assert doc.text[span.start : span.end] == span.metadata["text"]
    if id is not None:
        span.metadata["id"] = id
    doc.add_annotation(name=annotation_name, annotation=span)
    return span


def _add_relation(
    doc: Document,
    rel: BinaryRelation,
    annotation_name: str,
    id: Optional[str] = None,
) -> BinaryRelation:
    if id is not None:
        rel.metadata["id"] = id
    doc.add_annotation(name=annotation_name, annotation=rel)
    return rel


def get_doc1(
    entity_annotation_name: str,
    relation_annotation_name: str,
    sentence_annotation_name: str,
    **kwargs,
) -> Document:
    doc = Document(text=TEXT_01)
    for i, ent in enumerate([DOC1_ENTITY_JANE, DOC1_ENTITY_BERLIN, DOC1_ENTITY_KARL]):
        _add_span(
            doc=doc,
            span=ent,
            annotation_name=entity_annotation_name,
        )
    for i, sent in enumerate([DOC1_SENTENCE1]):
        _add_span(
            doc=doc,
            span=sent,
            annotation_name=sentence_annotation_name,
        )
    _add_relation(
        doc=doc,
        rel=DOC1_REL_LIVES_IN,
        annotation_name=relation_annotation_name,
    )
    return doc


def get_doc2(
    entity_annotation_name: str,
    relation_annotation_name: str,
    sentence_annotation_name: str,
) -> Document:
    doc = Document(text=TEXT_02)
    for i, ent in enumerate([DOC2_ENTITY_SEATTLE, DOC2_ENTITY_JENNY]):
        _add_span(
            doc=doc,
            span=ent,
            annotation_name=entity_annotation_name,
        )
    for i, sent in enumerate([DOC2_SENTENCE1, DOC2_SENTENCE2]):
        _add_span(
            doc=doc,
            span=sent,
            annotation_name=sentence_annotation_name,
        )

    _add_relation(
        doc=doc,
        rel=DOC2_REL_MAYOR_OF,
        annotation_name=relation_annotation_name,
    )
    return doc


def get_doc3(
    entity_annotation_name: str,
    relation_annotation_name: str,
    sentence_annotation_name: str,
) -> Document:
    doc = Document(text=TEXT_03)
    for i, ent in enumerate([DOC3_ENTITY_KARL, DOC3_ENTITY_BERLIN]):
        _add_span(
            doc=doc,
            span=ent,
            annotation_name=entity_annotation_name,
        )
    for i, sent in enumerate([DOC3_SENTENCE1]):
        _add_span(
            doc=doc,
            span=sent,
            annotation_name=sentence_annotation_name,
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
    documents = [get_doc1(**doc_kwargs), get_doc2(**doc_kwargs), get_doc3(**doc_kwargs)]
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
    assert (
        config["_type"]
        == "pytorch_ie.taskmodules.transformer_re_text_classification.TransformerRETextClassificationTaskModule"
    )
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
        assert target_labels == [DOC1_REL_LIVES_IN.label]

        encoding = task_encodings[1]
        assert encoding.has_target
        target_labels = [prepared_taskmodule.id_to_label[_id] for _id in encoding.target]
        assert target_labels == [DOC2_REL_MAYOR_OF.label]
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
        assert labels == [DOC1_REL_LIVES_IN.label, DOC2_REL_MAYOR_OF.label]
    else:
        assert targets is None


def test_unbatch_output(prepared_taskmodule, model_output):
    unbatched_outputs = prepared_taskmodule.unbatch_output(model_output)

    assert len(unbatched_outputs) == 2

    unbatched_output1 = unbatched_outputs[0]
    assert len(unbatched_output1["labels"]) == 1
    assert len(unbatched_output1["probabilities"]) == 1
    assert unbatched_output1["labels"][0] == DOC1_REL_LIVES_IN.label
    assert unbatched_output1["probabilities"][0] == pytest.approx(0.9999768733978271)

    unbatched_output2 = unbatched_outputs[1]
    assert len(unbatched_output2["labels"]) == 1
    assert len(unbatched_output2["probabilities"]) == 1
    assert unbatched_output2["labels"][0] == DOC2_REL_MAYOR_OF.label
    assert unbatched_output2["probabilities"][0] == pytest.approx(0.9999799728393555)


@pytest.mark.parametrize("inplace", [False, True])
def test_decode(prepared_taskmodule, documents, model_output, inplace):
    encodings = prepared_taskmodule.encode(documents, encode_target=False)
    unbatched_outputs = prepared_taskmodule.unbatch_output(model_output)
    decoded_documents = prepared_taskmodule.decode(
        encodings=encodings,
        decoded_outputs=unbatched_outputs,
        inplace=inplace,
        input_documents=documents,
    )

    assert len(decoded_documents) == 3
    if inplace:
        assert set(documents) == set(decoded_documents)
    else:
        for doc in decoded_documents:
            assert doc not in set(documents)

    predictions = decoded_documents[0].predictions("relations")
    assert len(predictions) == 1
    prediction = predictions[0]
    assert prediction.label == DOC1_REL_LIVES_IN.label
    assert prediction.head == DOC1_ENTITY_JANE
    assert prediction.tail == DOC1_ENTITY_BERLIN

    predictions = decoded_documents[1].predictions("relations")
    assert len(predictions) == 1
    prediction = predictions[0]
    assert prediction.label == DOC2_REL_MAYOR_OF.label
    assert prediction.head == DOC2_ENTITY_JENNY
    assert prediction.tail == DOC2_ENTITY_SEATTLE

    predictions = decoded_documents[2].predictions("relations")
    assert predictions is None


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


def test_enumerate_entity_pairs(prepared_taskmodule, documents):
    """
    This should return all combinations of entities.
    """
    document = documents[0]
    entities = document.span_annotations("entities")
    assert len(entities) == 3
    assert entities[0] == DOC1_ENTITY_JANE
    assert entities[1] == DOC1_ENTITY_BERLIN
    assert entities[2] == DOC1_ENTITY_KARL

    encoding = prepared_taskmodule._encode_text(
        document=document,  # partition=partition, add_special_tokens=add_special_tokens
    )
    assert (
        prepared_taskmodule.tokenizer.convert_ids_to_tokens(encoding.data["input_ids"])
        == DOC1_TOKENS
    )

    enumerated_entity_pairs = list(
        _enumerate_entity_pairs(
            entities=entities,  # partition=partition, relations=relations,
        )
    )
    assert len(enumerated_entity_pairs) == 6

    head, tail = enumerated_entity_pairs[0]
    assert head == DOC1_ENTITY_JANE
    assert tail == DOC1_ENTITY_BERLIN

    head, tail = enumerated_entity_pairs[1]
    assert head == DOC1_ENTITY_JANE
    assert tail == DOC1_ENTITY_KARL

    head, tail = enumerated_entity_pairs[2]
    assert head == DOC1_ENTITY_BERLIN
    assert tail == DOC1_ENTITY_JANE

    head, tail = enumerated_entity_pairs[3]
    assert head == DOC1_ENTITY_BERLIN
    assert tail == DOC1_ENTITY_KARL

    head, tail = enumerated_entity_pairs[4]
    assert head == DOC1_ENTITY_KARL
    assert tail == DOC1_ENTITY_JANE

    head, tail = enumerated_entity_pairs[5]
    assert head == DOC1_ENTITY_KARL
    assert tail == DOC1_ENTITY_BERLIN

    document = documents[1]
    entities = document.span_annotations("entities")
    assert len(entities) == 2
    assert entities[0] == DOC2_ENTITY_SEATTLE
    assert entities[1] == DOC2_ENTITY_JENNY

    encoding = prepared_taskmodule._encode_text(
        document=document,  # partition=partition, add_special_tokens=add_special_tokens
    )
    assert (
        prepared_taskmodule.tokenizer.convert_ids_to_tokens(encoding.data["input_ids"])
        == DOC2_TOKENS
    )

    enumerated_entity_pairs = list(
        _enumerate_entity_pairs(
            entities=entities,  # partition=partition, relations=relations,
        )
    )
    assert len(enumerated_entity_pairs) == 2

    head, tail = enumerated_entity_pairs[0]
    assert head == DOC2_ENTITY_SEATTLE
    assert tail == DOC2_ENTITY_JENNY

    head, tail = enumerated_entity_pairs[1]
    assert head == DOC2_ENTITY_JENNY
    assert tail == DOC2_ENTITY_SEATTLE

    document = documents[2]
    entities = document.span_annotations("entities")
    assert len(entities) == 2
    assert entities[0] == DOC3_ENTITY_KARL
    assert entities[1] == DOC3_ENTITY_BERLIN

    encoding = prepared_taskmodule._encode_text(
        document=document,  # partition=partition, add_special_tokens=add_special_tokens
    )
    assert (
        prepared_taskmodule.tokenizer.convert_ids_to_tokens(encoding.data["input_ids"])
        == DOC3_TOKENS
    )

    enumerated_entity_pairs = list(
        _enumerate_entity_pairs(
            entities=entities,  # partition=partition, relations=relations,
        )
    )
    assert len(enumerated_entity_pairs) == 2

    head, tail = enumerated_entity_pairs[0]
    assert head == DOC3_ENTITY_KARL
    assert tail == DOC3_ENTITY_BERLIN

    head, tail = enumerated_entity_pairs[1]
    assert head == DOC3_ENTITY_BERLIN
    assert tail == DOC3_ENTITY_KARL


def test_enumerate_entity_pairs_with_relations(prepared_taskmodule, documents):
    """
    This should return only combinations for which a relation exists.
    """
    document = documents[0]
    entities = document.span_annotations("entities")
    assert len(entities) == 3
    assert entities[0] == DOC1_ENTITY_JANE
    assert entities[1] == DOC1_ENTITY_BERLIN
    assert entities[2] == DOC1_ENTITY_KARL
    relations = document.relation_annotations("relations")
    assert len(relations) == 1
    relation = relations[0]
    assert relation == DOC1_REL_LIVES_IN
    assert relation.head == DOC1_ENTITY_JANE
    assert relation.tail == DOC1_ENTITY_BERLIN

    encoding = prepared_taskmodule._encode_text(
        document=document,  # partition=partition, add_special_tokens=add_special_tokens
    )
    assert (
        prepared_taskmodule.tokenizer.convert_ids_to_tokens(encoding.data["input_ids"])
        == DOC1_TOKENS
    )

    enumerated_entity_pairs = list(
        _enumerate_entity_pairs(
            entities=entities,
            relations=relations,
            # partition=partition,
        )
    )
    assert len(enumerated_entity_pairs) == 1
    head, tail = enumerated_entity_pairs[0]
    assert head == DOC1_ENTITY_JANE
    assert tail == DOC1_ENTITY_BERLIN

    document = documents[1]
    entities = document.span_annotations("entities")
    assert len(entities) == 2
    assert entities[0] == DOC2_ENTITY_SEATTLE
    assert entities[1] == DOC2_ENTITY_JENNY
    relations = document.relation_annotations("relations")
    assert len(relations) == 1
    relation = relations[0]
    assert relation == DOC2_REL_MAYOR_OF
    assert relation.head == DOC2_ENTITY_JENNY
    assert relation.tail == DOC2_ENTITY_SEATTLE

    encoding = prepared_taskmodule._encode_text(
        document=document,  # partition=partition, add_special_tokens=add_special_tokens
    )
    assert (
        prepared_taskmodule.tokenizer.convert_ids_to_tokens(encoding.data["input_ids"])
        == DOC2_TOKENS
    )

    enumerated_entity_pairs = list(
        _enumerate_entity_pairs(
            entities=entities,
            relations=relations,
            # partition=partition,
        )
    )
    assert len(enumerated_entity_pairs) == 1

    head, tail = enumerated_entity_pairs[0]
    assert head == DOC2_ENTITY_JENNY
    assert tail == DOC2_ENTITY_SEATTLE

    document = documents[2]
    entities = document.span_annotations("entities")
    assert len(entities) == 2
    assert entities[0] == DOC3_ENTITY_KARL
    assert entities[1] == DOC3_ENTITY_BERLIN
    relations = document.relation_annotations("relations")
    assert len(relations) == 0

    encoding = prepared_taskmodule._encode_text(
        document=document,  # partition=partition, add_special_tokens=add_special_tokens
    )
    assert (
        prepared_taskmodule.tokenizer.convert_ids_to_tokens(encoding.data["input_ids"])
        == DOC3_TOKENS
    )

    enumerated_entity_pairs = list(
        _enumerate_entity_pairs(
            entities=entities,
            relations=relations,  # partition=partition,
        )
    )
    assert len(enumerated_entity_pairs) == 0


def test_enumerate_entity_pairs_with_partitions(prepared_taskmodule, documents):
    """
    This should return only combinations with entities in the same sentence.
    """
    document = documents[0]
    entities = document.span_annotations("entities")
    assert len(entities) == 3
    assert entities[0] == DOC1_ENTITY_JANE
    assert entities[1] == DOC1_ENTITY_BERLIN
    assert entities[2] == DOC1_ENTITY_KARL
    sentences = document.span_annotations("sentences")
    assert len(sentences) == 1
    partition = sentences[0]
    assert partition == DOC1_SENTENCE1

    encoding = prepared_taskmodule._encode_text(
        document=document,
        partition=partition,  # add_special_tokens=add_special_tokens
    )
    assert prepared_taskmodule.tokenizer.convert_ids_to_tokens(encoding.data["input_ids"]) == [
        "[CLS]",
        "Jane",
        "lives",
        "in",
        "Berlin",
        ".",
        "[SEP]",
    ]

    enumerated_entity_pairs = list(
        _enumerate_entity_pairs(
            entities=entities,
            # relations=relations,
            partition=partition,
        )
    )
    assert len(enumerated_entity_pairs) == 2
    head, tail = enumerated_entity_pairs[0]
    assert head == DOC1_ENTITY_JANE
    assert tail == DOC1_ENTITY_BERLIN

    head, tail = enumerated_entity_pairs[1]
    assert head == DOC1_ENTITY_BERLIN
    assert tail == DOC1_ENTITY_JANE

    document = documents[1]
    entities = document.span_annotations("entities")
    assert len(entities) == 2
    assert entities[0] == DOC2_ENTITY_SEATTLE
    assert entities[1] == DOC2_ENTITY_JENNY
    sentences = document.span_annotations("sentences")
    assert len(sentences) == 2
    partition = sentences[0]
    assert partition == DOC2_SENTENCE1

    encoding = prepared_taskmodule._encode_text(
        document=document,
        partition=partition,  # add_special_tokens=add_special_tokens
    )
    assert prepared_taskmodule.tokenizer.convert_ids_to_tokens(encoding.data["input_ids"]) == [
        "[CLS]",
        "Seattle",
        "is",
        "a",
        "rainy",
        "city",
        ".",
        "[SEP]",
    ]

    enumerated_entity_pairs = list(
        _enumerate_entity_pairs(
            entities=entities,
            # relations=relations,
            partition=partition,
        )
    )
    assert len(enumerated_entity_pairs) == 0

    partition = sentences[1]
    assert partition == DOC2_SENTENCE2

    encoding = prepared_taskmodule._encode_text(
        document=document,
        partition=partition,  # add_special_tokens=add_special_tokens
    )
    assert prepared_taskmodule.tokenizer.convert_ids_to_tokens(encoding.data["input_ids"]) == [
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

    enumerated_entity_pairs = list(
        _enumerate_entity_pairs(
            entities=entities,
            # relations=relations,
            partition=partition,
        )
    )
    assert len(enumerated_entity_pairs) == 0

    document = documents[2]
    entities = document.span_annotations("entities")
    assert len(entities) == 2
    assert entities[0] == DOC3_ENTITY_KARL
    assert entities[1] == DOC3_ENTITY_BERLIN
    sentences = document.span_annotations("sentences")
    assert len(sentences) == 1
    partition = sentences[0]
    assert partition == DOC3_SENTENCE1

    encoding = prepared_taskmodule._encode_text(
        document=document,
        partition=partition,  # add_special_tokens=add_special_tokens
    )
    assert prepared_taskmodule.tokenizer.convert_ids_to_tokens(encoding.data["input_ids"]) == [
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

    enumerated_entity_pairs = list(
        _enumerate_entity_pairs(
            entities=entities,
            # relations=relations,
            partition=partition,
        )
    )
    assert len(enumerated_entity_pairs) == 2

    head, tail = enumerated_entity_pairs[0]
    assert head == DOC3_ENTITY_KARL
    assert tail == DOC3_ENTITY_BERLIN

    head, tail = enumerated_entity_pairs[1]
    assert head == DOC3_ENTITY_BERLIN
    assert tail == DOC3_ENTITY_KARL
