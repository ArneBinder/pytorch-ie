import re

import numpy
import pytest
import torch

from pytorch_ie.taskmodules import TransformerRETextClassificationTaskModule
from pytorch_ie.taskmodules.transformer_re_text_classification import _enumerate_entity_pairs


@pytest.fixture(scope="module")
def taskmodule():
    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = TransformerRETextClassificationTaskModule(
        tokenizer_name_or_path=tokenizer_name_or_path
    )
    return taskmodule


@pytest.fixture(scope="module", params=[False, True])
def taskmodule_optional_marker(request):
    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = TransformerRETextClassificationTaskModule(
        tokenizer_name_or_path=tokenizer_name_or_path, add_type_to_marker=request.param
    )
    return taskmodule


@pytest.fixture
def prepared_taskmodule(taskmodule, documents):
    taskmodule.prepare(documents)
    return taskmodule


@pytest.fixture
def prepared_taskmodule_optional_marker(taskmodule_optional_marker, documents):
    taskmodule_optional_marker.prepare(documents)
    return taskmodule_optional_marker


@pytest.fixture
def model_output():
    return {
        "logits": torch.from_numpy(
            numpy.log(
                [
                    # O, org:founded_by, per:employee_of, per:founder
                    [0.1, 0.6, 0.1, 0.2],
                    [0.5, 0.2, 0.2, 0.1],
                    [0.1, 0.2, 0.6, 0.1],
                    [0.1, 0.2, 0.2, 0.5],
                    [0.2, 0.4, 0.3, 0.1],
                    [0.5, 0.2, 0.2, 0.1],
                    [0.6, 0.1, 0.2, 0.1],
                    [0.5, 0.2, 0.2, 0.1],
                ]
            )
        ),
    }


def test_prepare(taskmodule_optional_marker, documents):
    taskmodule = taskmodule_optional_marker
    assert not taskmodule.is_prepared()
    taskmodule.prepare(documents)
    assert taskmodule.is_prepared()

    if taskmodule.add_type_to_marker:
        assert taskmodule.entity_labels == ["ORG", "PER"]
        assert taskmodule.argument_markers == {
            ("head", "start", "PER"): "[H:PER]",
            ("head", "end", "PER"): "[/H:PER]",
            ("tail", "start", "PER"): "[T:PER]",
            ("tail", "end", "PER"): "[/T:PER]",
            ("head", "start", "ORG"): "[H:ORG]",
            ("head", "end", "ORG"): "[/H:ORG]",
            ("tail", "start", "ORG"): "[T:ORG]",
            ("tail", "end", "ORG"): "[/T:ORG]",
        }
    else:
        assert taskmodule.entity_labels == []
        assert taskmodule.argument_markers == {
            ("head", "start"): "[H]",
            ("head", "end"): "[/H]",
            ("tail", "start"): "[T]",
            ("tail", "end"): "[/T]",
        }

    assert set(taskmodule.label_to_id.keys()) == {
        taskmodule.none_label,
        "per:employee_of",
        "org:founded_by",
        "per:founder",
    }
    assert [taskmodule.id_to_label[i] for i in range(4)] == [
        taskmodule.none_label,
        "org:founded_by",
        "per:employee_of",
        "per:founder",
    ]
    assert taskmodule.label_to_id[taskmodule.none_label] == 0


def test_config(prepared_taskmodule_optional_marker):
    prepared_taskmodule = prepared_taskmodule_optional_marker

    config = prepared_taskmodule._config()
    assert config["taskmodule_type"] == "TransformerRETextClassificationTaskModule"
    assert "label_to_id" in config
    assert config["label_to_id"] == {
        prepared_taskmodule.none_label: 0,
        "org:founded_by": 1,
        "per:employee_of": 2,
        "per:founder": 3,
    }

    if prepared_taskmodule.add_type_to_marker:
        assert config["entity_labels"] == ["ORG", "PER"]
    else:
        assert config["entity_labels"] == []


@pytest.mark.parametrize("encode_target", [False, True])
def test_encode(prepared_taskmodule_optional_marker, documents, encode_target):
    prepared_taskmodule = prepared_taskmodule_optional_marker

    task_encodings = prepared_taskmodule.encode(documents, encode_target=encode_target)

    if encode_target:
        assert len(task_encodings) == 7
    else:
        assert len(task_encodings) == 24

    encoding = task_encodings[0]

    tokens = prepared_taskmodule.tokenizer.convert_ids_to_tokens(encoding.inputs["input_ids"])

    if prepared_taskmodule.add_type_to_marker:
        tokens == [
            "[CLS]",
            "[H:PER]",
            "En",
            "##ti",
            "##ty",
            "A",
            "[/H:PER]",
            "works",
            "at",
            "[T:ORG]",
            "B",
            "[/T:ORG]",
            ".",
            "[SEP]",
        ]
    else:
        tokens == [
            "[CLS]",
            "[H]",
            "En",
            "##ti",
            "##ty",
            "A",
            "[/H]",
            "works",
            "at",
            "[T]",
            "B",
            "[/T]",
            ".",
            "[SEP]",
        ]

    if encode_target:
        assert encoding.targets == [2]
    else:
        assert not encoding.has_targets

        with pytest.raises(AssertionError, match=re.escape("task encoding has no target")):
            encoding.targets


@pytest.mark.parametrize("encode_target", [False, True])
def test_collate(prepared_taskmodule, documents, encode_target):
    documents = [documents[i] for i in [0, 1, 4]]

    encodings = prepared_taskmodule.encode(documents, encode_target=encode_target)

    if encode_target:
        assert len(encodings) == 4
        assert all([encoding.has_targets for encoding in encodings])
    else:
        assert len(encodings) == 8
        assert not any([encoding.has_targets for encoding in encodings])

    batch_encoding = prepared_taskmodule.collate(encodings)
    inputs, targets = batch_encoding

    assert "input_ids" in inputs
    assert "attention_mask" in inputs
    assert inputs["input_ids"].shape == inputs["attention_mask"].shape

    if encode_target:
        assert inputs["input_ids"].shape == (4, 21)
        assert len(targets) == 4
    else:
        assert inputs["input_ids"].shape == (8, 21)
        assert targets is None


def test_unbatch_output(prepared_taskmodule, model_output):
    unbatched_outputs = prepared_taskmodule.unbatch_output(model_output)

    assert len(unbatched_outputs) == 8

    labels = [
        "org:founded_by",
        "no_relation",
        "per:employee_of",
        "per:founder",
        "org:founded_by",
        "no_relation",
        "no_relation",
        "no_relation",
    ]
    probabilities = [0.6, 0.5, 0.6, 0.5, 0.4, 0.5, 0.6, 0.5]

    for output, label, probability in zip(unbatched_outputs, labels, probabilities):
        assert set(output.keys()) == {"labels", "probabilities"}
        assert output["labels"] == [label]
        assert output["probabilities"] == pytest.approx([probability])


@pytest.mark.parametrize("inplace", [False, True])
def test_decode(prepared_taskmodule, documents, model_output, inplace):
    documents = [documents[i] for i in [0, 1, 4]]

    encodings = prepared_taskmodule.encode(documents, encode_target=False)
    unbatched_outputs = prepared_taskmodule.unbatch_output(model_output)
    decoded_documents = prepared_taskmodule.decode(
        task_encodings=encodings,
        task_outputs=unbatched_outputs,
        documents_in_encode_order=documents,
        inplace=inplace,
    )

    assert len(decoded_documents) == len(documents)

    if inplace:
        assert {id(doc) for doc in decoded_documents} == {id(doc) for doc in documents}
    else:
        assert {id(doc) for doc in decoded_documents}.isdisjoint({id(doc) for doc in documents})

    expected_scores = [0.6, 0.5, 0.6, 0.5, 0.4, 0.5, 0.6, 0.5]
    i = 0
    for document in decoded_documents:
        for relation_expected, relation_decoded in zip(
            document["entities"], document["entities"].predictions
        ):
            assert relation_expected.start == relation_decoded.start
            assert relation_expected.end == relation_decoded.end
            assert relation_expected.label == relation_decoded.label
            assert expected_scores[i] == pytest.approx(relation_decoded.score)
            i += 1

    if not inplace:
        for document in documents:
            assert not document["relations"].predictions


# def test_encode_input_with_partitions(prepared_taskmodule_optional_marker, documents):
#     prepared_taskmodule_with_partitions = copy.deepcopy(prepared_taskmodule_optional_marker)
#     prepared_taskmodule_with_partitions.partition_annotation = "sentences"
#     input_encoding, metadata, new_documents = prepared_taskmodule_with_partitions.encode_input(
#         documents, is_training=True
#     )
#     assert len(input_encoding) == 1
#     assert new_documents is not None
#     assert len(new_documents) == 1
#     encoding = input_encoding[0]
#     if prepared_taskmodule_with_partitions.add_type_to_marker:
#         assert prepared_taskmodule_optional_marker.tokenizer.convert_ids_to_tokens(
#             encoding["input_ids"]
#         ) == [
#             "[CLS]",
#             "[H:person]",
#             "Jane",
#             "[/H:person]",
#             "lives",
#             "in",
#             "[T:city]",
#             "Berlin",
#             "[/T:city]",
#             ".",
#             "[SEP]",
#         ]
#     else:
#         assert prepared_taskmodule_optional_marker.tokenizer.convert_ids_to_tokens(
#             encoding["input_ids"]
#         ) == [
#             "[CLS]",
#             "[H]",
#             "Jane",
#             "[/H]",
#             "lives",
#             "in",
#             "[T]",
#             "Berlin",
#             "[/T]",
#             ".",
#             "[SEP]",
#         ]


# def test_encode_with_windowing(prepared_taskmodule_optional_marker, documents):
#     prepared_taskmodule_with_windowing = copy.deepcopy(prepared_taskmodule_optional_marker)
#     prepared_taskmodule_with_windowing.max_window = 10
#     task_encodings = prepared_taskmodule_with_windowing.encode(documents, encode_target=False)
#     assert len(task_encodings) == 2

#     # just check the first entry
#     encoding = task_encodings[0]
#     document = documents[0]
#     assert encoding.document == document
#     assert "input_ids" in encoding.input
#     assert len(encoding.input["input_ids"]) <= prepared_taskmodule_with_windowing.max_window
#     if prepared_taskmodule_with_windowing.add_type_to_marker:
#         assert prepared_taskmodule_with_windowing.tokenizer.convert_ids_to_tokens(
#             encoding.input["input_ids"]
#         ) == [
#             "[CLS]",
#             "[H:person]",
#             "Jane",
#             "[/H:person]",
#             "lives",
#             "in",
#             "[T:city]",
#             "Berlin",
#             "[/T:city]",
#             "[SEP]",
#         ]
#     else:
#         assert prepared_taskmodule_with_windowing.tokenizer.convert_ids_to_tokens(
#             encoding.input["input_ids"]
#         ) == ["[CLS]", "[H]", "Jane", "[/H]", "lives", "in", "[T]", "Berlin", "[/T]", "[SEP]"]


# def test_save_load(tmp_path, prepared_taskmodule):
#     path = os.path.join(tmp_path, "taskmodule")
#     prepared_taskmodule.save_pretrained(path)
#     loaded_taskmodule = TransformerRETextClassificationTaskModule.from_pretrained(path)
#     assert loaded_taskmodule.is_prepared()
#     assert loaded_taskmodule.argument_markers == prepared_taskmodule.argument_markers


# def test_enumerate_entity_pairs(prepared_taskmodule, documents):
#     """
#     This should return all combinations of entities.
#     """
#     document = documents[0]
#     entities = document.annotations.spans["entities"]
#     assert len(entities) == 3
#     assert entities[0] == DOC1_ENTITY_JANE
#     assert entities[1] == DOC1_ENTITY_BERLIN
#     assert entities[2] == DOC1_ENTITY_KARL

#     encoding = prepared_taskmodule._encode_text(
#         document=document,  # partition=partition, add_special_tokens=add_special_tokens
#     )
#     assert (
#         prepared_taskmodule.tokenizer.convert_ids_to_tokens(encoding.data["input_ids"])
#         == DOC1_TOKENS
#     )

#     enumerated_entity_pairs = list(
#         _enumerate_entity_pairs(
#             entities=entities,  # partition=partition, relations=relations,
#         )
#     )
#     assert len(enumerated_entity_pairs) == 6

#     head, tail = enumerated_entity_pairs[0]
#     assert head == DOC1_ENTITY_JANE
#     assert tail == DOC1_ENTITY_BERLIN

#     head, tail = enumerated_entity_pairs[1]
#     assert head == DOC1_ENTITY_JANE
#     assert tail == DOC1_ENTITY_KARL

#     head, tail = enumerated_entity_pairs[2]
#     assert head == DOC1_ENTITY_BERLIN
#     assert tail == DOC1_ENTITY_JANE

#     head, tail = enumerated_entity_pairs[3]
#     assert head == DOC1_ENTITY_BERLIN
#     assert tail == DOC1_ENTITY_KARL

#     head, tail = enumerated_entity_pairs[4]
#     assert head == DOC1_ENTITY_KARL
#     assert tail == DOC1_ENTITY_JANE

#     head, tail = enumerated_entity_pairs[5]
#     assert head == DOC1_ENTITY_KARL
#     assert tail == DOC1_ENTITY_BERLIN

#     document = documents[1]
#     entities = document.annotations.spans["entities"]
#     assert len(entities) == 2
#     assert entities[0] == DOC2_ENTITY_SEATTLE
#     assert entities[1] == DOC2_ENTITY_JENNY

#     encoding = prepared_taskmodule._encode_text(
#         document=document,  # partition=partition, add_special_tokens=add_special_tokens
#     )
#     assert (
#         prepared_taskmodule.tokenizer.convert_ids_to_tokens(encoding.data["input_ids"])
#         == DOC2_TOKENS
#     )

#     enumerated_entity_pairs = list(
#         _enumerate_entity_pairs(
#             entities=entities,  # partition=partition, relations=relations,
#         )
#     )
#     assert len(enumerated_entity_pairs) == 2

#     head, tail = enumerated_entity_pairs[0]
#     assert head == DOC2_ENTITY_SEATTLE
#     assert tail == DOC2_ENTITY_JENNY

#     head, tail = enumerated_entity_pairs[1]
#     assert head == DOC2_ENTITY_JENNY
#     assert tail == DOC2_ENTITY_SEATTLE

#     document = documents[2]
#     entities = document.annotations.spans["entities"]
#     assert len(entities) == 2
#     assert entities[0] == DOC3_ENTITY_KARL
#     assert entities[1] == DOC3_ENTITY_BERLIN

#     encoding = prepared_taskmodule._encode_text(
#         document=document,  # partition=partition, add_special_tokens=add_special_tokens
#     )
#     assert (
#         prepared_taskmodule.tokenizer.convert_ids_to_tokens(encoding.data["input_ids"])
#         == DOC3_TOKENS
#     )

#     enumerated_entity_pairs = list(
#         _enumerate_entity_pairs(
#             entities=entities,  # partition=partition, relations=relations,
#         )
#     )
#     assert len(enumerated_entity_pairs) == 2

#     head, tail = enumerated_entity_pairs[0]
#     assert head == DOC3_ENTITY_KARL
#     assert tail == DOC3_ENTITY_BERLIN

#     head, tail = enumerated_entity_pairs[1]
#     assert head == DOC3_ENTITY_BERLIN
#     assert tail == DOC3_ENTITY_KARL


# def test_enumerate_entity_pairs_with_relations(prepared_taskmodule, documents):
#     """
#     This should return only combinations for which a relation exists.
#     """
#     document = documents[0]
#     entities = document.annotations.spans["entities"]
#     assert len(entities) == 3
#     assert entities[0] == DOC1_ENTITY_JANE
#     assert entities[1] == DOC1_ENTITY_BERLIN
#     assert entities[2] == DOC1_ENTITY_KARL
#     relations = document.annotations.binary_relations["relations"]
#     assert len(relations) == 1
#     relation = relations[0]
#     assert relation == DOC1_REL_LIVES_IN
#     assert relation.head == DOC1_ENTITY_JANE
#     assert relation.tail == DOC1_ENTITY_BERLIN

#     encoding = prepared_taskmodule._encode_text(
#         document=document,  # partition=partition, add_special_tokens=add_special_tokens
#     )
#     assert (
#         prepared_taskmodule.tokenizer.convert_ids_to_tokens(encoding.data["input_ids"])
#         == DOC1_TOKENS
#     )

#     enumerated_entity_pairs = list(
#         _enumerate_entity_pairs(
#             entities=entities,
#             relations=relations,
#             # partition=partition,
#         )
#     )
#     assert len(enumerated_entity_pairs) == 1
#     head, tail = enumerated_entity_pairs[0]
#     assert head == DOC1_ENTITY_JANE
#     assert tail == DOC1_ENTITY_BERLIN

#     document = documents[1]
#     entities = document.annotations.spans["entities"]
#     assert len(entities) == 2
#     assert entities[0] == DOC2_ENTITY_SEATTLE
#     assert entities[1] == DOC2_ENTITY_JENNY
#     relations = document.annotations.binary_relations["relations"]
#     assert len(relations) == 1
#     relation = relations[0]
#     assert relation == DOC2_REL_MAYOR_OF
#     assert relation.head == DOC2_ENTITY_JENNY
#     assert relation.tail == DOC2_ENTITY_SEATTLE

#     encoding = prepared_taskmodule._encode_text(
#         document=document,  # partition=partition, add_special_tokens=add_special_tokens
#     )
#     assert (
#         prepared_taskmodule.tokenizer.convert_ids_to_tokens(encoding.data["input_ids"])
#         == DOC2_TOKENS
#     )

#     enumerated_entity_pairs = list(
#         _enumerate_entity_pairs(
#             entities=entities,
#             relations=relations,
#             # partition=partition,
#         )
#     )
#     assert len(enumerated_entity_pairs) == 1

#     head, tail = enumerated_entity_pairs[0]
#     assert head == DOC2_ENTITY_JENNY
#     assert tail == DOC2_ENTITY_SEATTLE

#     document = documents[2]
#     entities = document.annotations.spans["entities"]
#     assert len(entities) == 2
#     assert entities[0] == DOC3_ENTITY_KARL
#     assert entities[1] == DOC3_ENTITY_BERLIN
#     relations = document.annotations.binary_relations["relations"]
#     assert len(relations) == 0

#     encoding = prepared_taskmodule._encode_text(
#         document=document,  # partition=partition, add_special_tokens=add_special_tokens
#     )
#     assert (
#         prepared_taskmodule.tokenizer.convert_ids_to_tokens(encoding.data["input_ids"])
#         == DOC3_TOKENS
#     )

#     enumerated_entity_pairs = list(
#         _enumerate_entity_pairs(
#             entities=entities,
#             relations=relations,  # partition=partition,
#         )
#     )
#     assert len(enumerated_entity_pairs) == 0


# def test_enumerate_entity_pairs_with_partitions(prepared_taskmodule, documents):
#     """
#     This should return only combinations with entities in the same sentence.
#     """
#     document = documents[0]
#     entities = document.annotations.spans["entities"]
#     assert len(entities) == 3
#     assert entities[0] == DOC1_ENTITY_JANE
#     assert entities[1] == DOC1_ENTITY_BERLIN
#     assert entities[2] == DOC1_ENTITY_KARL
#     sentences = document.annotations.spans["sentences"]
#     assert len(sentences) == 1
#     partition = sentences[0]
#     assert partition == DOC1_SENTENCE1

#     encoding = prepared_taskmodule._encode_text(
#         document=document,
#         partition=partition,  # add_special_tokens=add_special_tokens
#     )
#     assert prepared_taskmodule.tokenizer.convert_ids_to_tokens(encoding.data["input_ids"]) == [
#         "[CLS]",
#         "Jane",
#         "lives",
#         "in",
#         "Berlin",
#         ".",
#         "[SEP]",
#     ]

#     enumerated_entity_pairs = list(
#         _enumerate_entity_pairs(
#             entities=entities,
#             # relations=relations,
#             partition=partition,
#         )
#     )
#     assert len(enumerated_entity_pairs) == 2
#     head, tail = enumerated_entity_pairs[0]
#     assert head == DOC1_ENTITY_JANE
#     assert tail == DOC1_ENTITY_BERLIN

#     head, tail = enumerated_entity_pairs[1]
#     assert head == DOC1_ENTITY_BERLIN
#     assert tail == DOC1_ENTITY_JANE

#     document = documents[1]
#     entities = document.annotations.spans["entities"]
#     assert len(entities) == 2
#     assert entities[0] == DOC2_ENTITY_SEATTLE
#     assert entities[1] == DOC2_ENTITY_JENNY
#     sentences = document.annotations.spans["sentences"]
#     assert len(sentences) == 2
#     partition = sentences[0]
#     assert partition == DOC2_SENTENCE1

#     encoding = prepared_taskmodule._encode_text(
#         document=document,
#         partition=partition,  # add_special_tokens=add_special_tokens
#     )
#     assert prepared_taskmodule.tokenizer.convert_ids_to_tokens(encoding.data["input_ids"]) == [
#         "[CLS]",
#         "Seattle",
#         "is",
#         "a",
#         "rainy",
#         "city",
#         ".",
#         "[SEP]",
#     ]

#     enumerated_entity_pairs = list(
#         _enumerate_entity_pairs(
#             entities=entities,
#             # relations=relations,
#             partition=partition,
#         )
#     )
#     assert len(enumerated_entity_pairs) == 0

#     partition = sentences[1]
#     assert partition == DOC2_SENTENCE2

#     encoding = prepared_taskmodule._encode_text(
#         document=document,
#         partition=partition,  # add_special_tokens=add_special_tokens
#     )
#     assert prepared_taskmodule.tokenizer.convert_ids_to_tokens(encoding.data["input_ids"]) == [
#         "[CLS]",
#         "Jenny",
#         "Du",
#         "##rka",
#         "##n",
#         "is",
#         "the",
#         "city",
#         "'",
#         "s",
#         "mayor",
#         ".",
#         "[SEP]",
#     ]

#     enumerated_entity_pairs = list(
#         _enumerate_entity_pairs(
#             entities=entities,
#             # relations=relations,
#             partition=partition,
#         )
#     )
#     assert len(enumerated_entity_pairs) == 0

#     document = documents[2]
#     entities = document.annotations.spans["entities"]
#     assert len(entities) == 2
#     assert entities[0] == DOC3_ENTITY_KARL
#     assert entities[1] == DOC3_ENTITY_BERLIN
#     sentences = document.annotations.spans["sentences"]
#     assert len(sentences) == 1
#     partition = sentences[0]
#     assert partition == DOC3_SENTENCE1

#     encoding = prepared_taskmodule._encode_text(
#         document=document,
#         partition=partition,  # add_special_tokens=add_special_tokens
#     )
#     assert prepared_taskmodule.tokenizer.convert_ids_to_tokens(encoding.data["input_ids"]) == [
#         "[CLS]",
#         "Karl",
#         "enjoys",
#         "sunny",
#         "days",
#         "in",
#         "Berlin",
#         ".",
#         "[SEP]",
#     ]

#     enumerated_entity_pairs = list(
#         _enumerate_entity_pairs(
#             entities=entities,
#             # relations=relations,
#             partition=partition,
#         )
#     )
#     assert len(enumerated_entity_pairs) == 2

#     head, tail = enumerated_entity_pairs[0]
#     assert head == DOC3_ENTITY_KARL
#     assert tail == DOC3_ENTITY_BERLIN

#     head, tail = enumerated_entity_pairs[1]
#     assert head == DOC3_ENTITY_BERLIN
#     assert tail == DOC3_ENTITY_KARL
