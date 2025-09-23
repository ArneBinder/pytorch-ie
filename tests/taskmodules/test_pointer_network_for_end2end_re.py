import logging
import pickle
from dataclasses import asdict, dataclass
from typing import Dict, List, Set

import pytest
import torch
from pie_core import AnnotationLayer, Document, annotation_field
from pie_documents.annotations import BinaryRelation, LabeledSpan
from pie_documents.documents import TextBasedDocument
from transformers import LogitsProcessorList

from pytorch_ie.taskmodules import PointerNetworkTaskModuleForEnd2EndRE
from pytorch_ie.taskmodules.pointer_network.logits_processor import (
    FinitizeLogitsProcessor,
    PrefixConstrainedLogitsProcessorWithMaximum,
)
from pytorch_ie.taskmodules.pointer_network_for_end2end_re import LabelsAndOptionalConstraints

logger = logging.getLogger(__name__)

DUMP_FIXTURE_DATA = False


def _config_to_str(cfg: Dict[str, str]) -> str:
    result = "-".join([f"{k}={cfg[k]}" for k in sorted(cfg)])
    return result


CONFIGS = [{}, {"partition_layer_name": "sentences"}]
CONFIG_DICT = {_config_to_str(cfg): cfg for cfg in CONFIGS}


@pytest.fixture(scope="module", params=CONFIG_DICT.keys())
def config_str(request):
    return request.param


@pytest.fixture(scope="module")
def config(config_str):
    return CONFIG_DICT[config_str]


@dataclass
class ExampleDocument(TextBasedDocument):
    entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationLayer[BinaryRelation] = annotation_field(target="entities")
    sentences: AnnotationLayer[LabeledSpan] = annotation_field(target="text")


@pytest.fixture(scope="module")
def document():
    doc = ExampleDocument(text="This is a dummy text about nothing. Trust me.")
    span1 = LabeledSpan(start=10, end=20, label="content")
    span2 = LabeledSpan(start=27, end=34, label="topic")
    span3 = LabeledSpan(start=42, end=44, label="person")
    doc.entities.extend([span1, span2, span3])
    assert str(span1) == "dummy text"
    assert str(span2) == "nothing"
    assert str(span3) == "me"
    rel = BinaryRelation(head=span1, tail=span2, label="is_about")
    doc.relations.append(rel)
    assert str(rel.label) == "is_about"
    assert str(rel.head) == "dummy text"
    assert str(rel.tail) == "nothing"

    no_rel = BinaryRelation(head=span1, tail=span3, label="no_relation")
    doc.relations.append(no_rel)
    assert str(no_rel.label) == "no_relation"
    assert str(no_rel.head) == "dummy text"
    assert str(no_rel.tail) == "me"

    sent1 = LabeledSpan(start=0, end=35, label="1")
    sent2 = LabeledSpan(start=36, end=45, label="2")
    doc.sentences.extend([sent1, sent2])
    assert str(sent1) == "This is a dummy text about nothing."
    assert str(sent2) == "Trust me."
    return doc


def test_document(document):
    spans = document.entities
    assert len(spans) == 3
    assert (str(spans[0]), spans[0].label) == ("dummy text", "content")
    assert (str(spans[1]), spans[1].label) == ("nothing", "topic")
    assert (str(spans[2]), spans[2].label) == ("me", "person")
    relations = document.relations
    assert len(relations) == 2
    assert (str(relations[0].head), relations[0].label, str(relations[0].tail)) == (
        "dummy text",
        "is_about",
        "nothing",
    )
    assert (str(relations[1].head), relations[1].label, str(relations[1].tail)) == (
        "dummy text",
        "no_relation",
        "me",
    )
    sentences = document.sentences
    assert len(sentences) == 2
    assert str(sentences[0]) == "This is a dummy text about nothing."
    assert str(sentences[1]) == "Trust me."


@pytest.fixture(scope="module")
def taskmodule(document, config):
    taskmodule = PointerNetworkTaskModuleForEnd2EndRE(
        tokenizer_name_or_path="facebook/bart-base",
        relation_layer_name="relations",
        exclude_labels_per_layer={"relations": ["no_relation"]},
        annotation_field_mapping={
            "entities": "labeled_spans",
            "relations": "binary_relations",
        },
        create_constraints=True,
        tokenizer_kwargs={"strict_span_conversion": False},
        **config,
    )

    taskmodule.prepare(documents=[document])
    return taskmodule


def test_taskmodule(taskmodule):
    assert taskmodule.is_prepared
    assert taskmodule.prepared_attributes == {
        "labels_per_layer": {
            "entities": ["content", "person", "topic"],
            "relations": ["is_about"],
        },
    }
    assert taskmodule.layer_names == ["entities", "relations"]
    assert taskmodule.special_targets == ["<s>", "</s>"]
    assert taskmodule.labels == ["none", "content", "person", "topic", "is_about"]
    assert taskmodule.targets == [
        "<s>",
        "</s>",
        "none",
        "content",
        "person",
        "topic",
        "is_about",
    ]
    assert taskmodule.bos_id == 0
    assert taskmodule.eos_id == 1
    assert taskmodule.none_id == 2
    assert taskmodule.span_ids == [3, 4, 5]
    assert taskmodule.relation_ids == [6]
    assert taskmodule.label2id == {
        "content": 3,
        "is_about": 6,
        "none": 2,
        "person": 4,
        "topic": 5,
    }
    assert taskmodule.label_embedding_weight_mapping == {
        50265: [45260],
        50266: [39763],
        50267: [354, 1215, 9006],
        50268: [5970],
        50269: [10166],
    }
    assert taskmodule.target_tokens == [
        "<s>",
        "</s>",
        "<<none>>",
        "<<content>>",
        "<<person>>",
        "<<topic>>",
        "<<is_about>>",
    ]
    assert taskmodule.target_token_ids == [0, 2, 50266, 50269, 50268, 50265, 50267]


def test_taskmodule_with_wrong_annotation_field_mapping():
    with pytest.raises(ValueError) as exc_info:
        PointerNetworkTaskModuleForEnd2EndRE(
            tokenizer_name_or_path="facebook/bart-base",
            relation_layer_name="relations",
            annotation_field_mapping={
                "entities": "labeled_spans",
                "sentences": "labeled_spans",
            },
        )
    assert str(exc_info.value) == (
        "inverted annotation_field_mapping is not unique. annotation_field_mapping: "
        "{'entities': 'labeled_spans', 'sentences': 'labeled_spans'}"
    )


def test_prepared_config(taskmodule, config):
    if config == {}:
        assert taskmodule._config() == {
            "taskmodule_type": "PointerNetworkTaskModuleForEnd2EndRE",
            "relation_layer_name": "relations",
            "symmetric_relations": None,
            "none_label": "none",
            "loop_dummy_relation_name": "loop",
            "labels_per_layer": {
                "entities": ["content", "person", "topic"],
                "relations": ["is_about"],
            },
            "exclude_labels_per_layer": {"relations": ["no_relation"]},
            "create_constraints": True,
            "document_type": "pie_documents.documents.TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions",
            "tokenized_document_type": "pie_documents.documents.TokenDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions",
            "tokenizer_name_or_path": "facebook/bart-base",
            "tokenizer_init_kwargs": None,
            "tokenizer_kwargs": {"strict_span_conversion": False},
            "partition_layer_name": None,
            "add_reversed_relations": False,
            "annotation_field_mapping": {
                "entities": "labeled_spans",
                "relations": "binary_relations",
            },
            "constrained_generation": False,
            "label_tokens": None,
            "label_representations": None,
            "log_first_n_examples": None,
        }
    elif config == {"partition_layer_name": "sentences"}:
        assert taskmodule._config() == {
            "taskmodule_type": "PointerNetworkTaskModuleForEnd2EndRE",
            "relation_layer_name": "relations",
            "symmetric_relations": None,
            "none_label": "none",
            "loop_dummy_relation_name": "loop",
            "labels_per_layer": {
                "entities": ["content", "person", "topic"],
                "relations": ["is_about"],
            },
            "exclude_labels_per_layer": {"relations": ["no_relation"]},
            "create_constraints": True,
            "document_type": "pie_documents.documents.TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions",
            "tokenized_document_type": "pie_documents.documents.TokenDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions",
            "tokenizer_name_or_path": "facebook/bart-base",
            "tokenizer_init_kwargs": None,
            "tokenizer_kwargs": {"strict_span_conversion": False},
            "partition_layer_name": "sentences",
            "add_reversed_relations": False,
            "annotation_field_mapping": {
                "entities": "labeled_spans",
                "relations": "binary_relations",
            },
            "constrained_generation": False,
            "label_tokens": None,
            "label_representations": None,
            "log_first_n_examples": None,
        }
    else:
        raise Exception(f"unknown config: {config}")


@pytest.fixture()
def task_encoding_without_target(taskmodule, document):
    return taskmodule.encode_input(document)[0]


def test_add_reversed_relation_labels():
    taskmodule = PointerNetworkTaskModuleForEnd2EndRE(
        tokenizer_name_or_path="facebook/bart-base",
        symmetric_relations=["symmetric_relation"],
    )

    labels = ["is_about", "symmetric_relation"]
    labels_with_reversed = taskmodule.add_reversed_relation_labels(labels)
    assert labels_with_reversed == {"is_about", "is_about_reversed", "symmetric_relation"}


def test_reverse_relation():
    taskmodule = PointerNetworkTaskModuleForEnd2EndRE(
        tokenizer_name_or_path="facebook/bart-base",
        symmetric_relations=["symmetric_relation"],
    )

    rel = BinaryRelation(
        head=LabeledSpan(start=10, end=20, label="content"),
        tail=LabeledSpan(start=27, end=34, label="topic"),
        label="is_about",
    )
    reversed_relation = taskmodule.reverse_relation(relation=rel)
    assert reversed_relation == BinaryRelation(
        head=LabeledSpan(start=27, end=34, label="topic", score=1.0),
        tail=LabeledSpan(start=10, end=20, label="content", score=1.0),
        label="is_about_reversed",
        score=1.0,
    )

    sym_rel = BinaryRelation(
        head=LabeledSpan(start=10, end=20, label="content"),
        tail=LabeledSpan(start=27, end=34, label="topic"),
        label="symmetric_relation",
    )
    reversed_sym_rel = taskmodule.reverse_relation(relation=sym_rel)
    assert reversed_sym_rel == BinaryRelation(
        head=LabeledSpan(start=27, end=34, label="topic", score=1.0),
        tail=LabeledSpan(start=10, end=20, label="content", score=1.0),
        label="symmetric_relation",
        score=1.0,
    )


def test_unreverse_relation():
    taskmodule = PointerNetworkTaskModuleForEnd2EndRE(
        tokenizer_name_or_path="facebook/bart-base",
        symmetric_relations=["symmetric_relation"],
    )

    # nothing should change because the relation is not reversed
    rel = BinaryRelation(
        head=LabeledSpan(start=10, end=20, label="content"),
        tail=LabeledSpan(start=27, end=34, label="topic"),
        label="is_about",
    )
    same_rel = taskmodule.unreverse_relation(relation=rel)
    assert same_rel == rel

    # the relation is reversed, so it should be un-reversed
    reversed_rel = BinaryRelation(
        head=LabeledSpan(start=10, end=20, label="content"),
        tail=LabeledSpan(start=27, end=34, label="topic"),
        label="is_about_reversed",
    )
    unreversed_relation = taskmodule.unreverse_relation(relation=reversed_rel)
    assert unreversed_relation == BinaryRelation(
        head=LabeledSpan(start=27, end=34, label="topic", score=1.0),
        tail=LabeledSpan(start=10, end=20, label="content", score=1.0),
        label="is_about",
        score=1.0,
    )

    # nothing should change because the relation is symmetric and already ordered (head < tail)
    ordered_sym_rel = BinaryRelation(
        head=LabeledSpan(start=10, end=20, label="content"),
        tail=LabeledSpan(start=27, end=34, label="topic"),
        label="symmetric_relation",
    )
    unreversed_ordered_sym_rel = taskmodule.unreverse_relation(relation=ordered_sym_rel)
    assert ordered_sym_rel == unreversed_ordered_sym_rel

    # the relation is symmetric and unordered (head > tail), so it should be un-reversed
    unordered_sym_rel = BinaryRelation(
        head=LabeledSpan(start=27, end=34, label="topic"),
        tail=LabeledSpan(start=10, end=20, label="content"),
        label="symmetric_relation",
    )
    unreversed_unordered_sym_rel = taskmodule.unreverse_relation(relation=unordered_sym_rel)
    assert unreversed_unordered_sym_rel == BinaryRelation(
        head=LabeledSpan(start=10, end=20, label="content", score=1.0),
        tail=LabeledSpan(start=27, end=34, label="topic", score=1.0),
        label="symmetric_relation",
        score=1.0,
    )


@pytest.fixture(params=[False, True])
def taskmodule_with_reversed_relations(document, request) -> PointerNetworkTaskModuleForEnd2EndRE:
    is_about_is_symmetric = request.param
    taskmodule = PointerNetworkTaskModuleForEnd2EndRE(
        tokenizer_name_or_path="facebook/bart-base",
        relation_layer_name="relations",
        exclude_labels_per_layer={"relations": ["no_relation"]},
        annotation_field_mapping={
            "entities": "labeled_spans",
            "relations": "binary_relations",
        },
        create_constraints=True,
        tokenizer_kwargs={"strict_span_conversion": False},
        add_reversed_relations=True,
        symmetric_relations=["is_about"] if is_about_is_symmetric else None,
    )

    taskmodule.prepare(documents=[document])
    assert taskmodule.is_prepared
    if is_about_is_symmetric:
        assert taskmodule.prepared_attributes == {
            "labels_per_layer": {
                "entities": ["content", "person", "topic"],
                "relations": ["is_about"],
            }
        }
    else:
        assert taskmodule.prepared_attributes == {
            "labels_per_layer": {
                "entities": ["content", "person", "topic"],
                "relations": ["is_about", "is_about_reversed"],
            }
        }

    return taskmodule


def test_encode_with_add_reversed_relations(taskmodule_with_reversed_relations, document):
    task_encodings = taskmodule_with_reversed_relations.encode(document, encode_target=True)
    assert len(task_encodings) == 1
    task_encoding = task_encodings[0]
    assert task_encoding is not None
    assert asdict(task_encoding.inputs) == {
        "input_ids": [0, 713, 16, 10, 34759, 2788, 59, 1085, 4, 3101, 162, 4, 2],
        "attention_mask": [1] * 13,
    }
    tokens = taskmodule_with_reversed_relations.tokenizer.convert_ids_to_tokens(
        task_encoding.inputs.input_ids
    )
    assert tokens == [
        "<s>",
        "This",
        "Ġis",
        "Ġa",
        "Ġdummy",
        "Ġtext",
        "Ġabout",
        "Ġnothing",
        ".",
        "ĠTrust",
        "Ġme",
        ".",
        "</s>",
    ]
    if "is_about" in taskmodule_with_reversed_relations.symmetric_relations:
        decoded_annotations, statistics = taskmodule_with_reversed_relations.decode_annotations(
            task_encoding.targets
        )
        assert decoded_annotations == {
            "entities": [
                LabeledSpan(start=4, end=6, label="content", score=1.0),
                LabeledSpan(start=7, end=8, label="topic", score=1.0),
                LabeledSpan(start=10, end=11, label="person", score=1.0),
            ],
            "relations": [
                BinaryRelation(
                    head=LabeledSpan(start=4, end=6, label="content", score=1.0),
                    tail=LabeledSpan(start=7, end=8, label="topic", score=1.0),
                    label="is_about",
                    score=1.0,
                ),
                BinaryRelation(
                    head=LabeledSpan(start=7, end=8, label="topic", score=1.0),
                    tail=LabeledSpan(start=4, end=6, label="content", score=1.0),
                    label="is_about",
                    score=1.0,
                ),
            ],
        }
    else:
        decoded_annotations, statistics = taskmodule_with_reversed_relations.decode_annotations(
            task_encoding.targets
        )
        assert decoded_annotations == {
            "entities": [
                LabeledSpan(start=4, end=6, label="content", score=1.0),
                LabeledSpan(start=7, end=8, label="topic", score=1.0),
                LabeledSpan(start=10, end=11, label="person", score=1.0),
            ],
            "relations": [
                BinaryRelation(
                    head=LabeledSpan(start=4, end=6, label="content", score=1.0),
                    tail=LabeledSpan(start=7, end=8, label="topic", score=1.0),
                    label="is_about",
                    score=1.0,
                ),
                BinaryRelation(
                    head=LabeledSpan(start=7, end=8, label="topic", score=1.0),
                    tail=LabeledSpan(start=4, end=6, label="content", score=1.0),
                    label="is_about_reversed",
                    score=1.0,
                ),
            ],
        }


def test_encode_with_add_reversed_relations_already_exists(caplog):
    doc = ExampleDocument(text="This is a dummy text about nothing. Trust me.")
    doc.entities.append(LabeledSpan(start=10, end=20, label="content"))
    doc.entities.append(LabeledSpan(start=27, end=34, label="topic"))
    doc.relations.append(
        BinaryRelation(head=doc.entities[0], tail=doc.entities[1], label="is_about")
    )
    doc.relations.append(
        BinaryRelation(head=doc.entities[1], tail=doc.entities[0], label="is_about")
    )

    taskmodule = PointerNetworkTaskModuleForEnd2EndRE(
        tokenizer_name_or_path="facebook/bart-base",
        relation_layer_name="relations",
        annotation_field_mapping={
            "entities": "labeled_spans",
            "relations": "binary_relations",
        },
        add_reversed_relations=True,
        symmetric_relations=["is_about"],
    )
    taskmodule.prepare(documents=[doc])

    with caplog.at_level(logging.WARNING):
        task_encodings = taskmodule.encode(doc, encode_target=True)
    assert len(caplog.messages) == 0
    assert len(task_encodings) == 1
    task_encoding = task_encodings[0]

    decoded_annotations, statistics = taskmodule.decode_annotations(task_encoding.targets)
    assert decoded_annotations == {
        "entities": [
            LabeledSpan(start=4, end=6, label="content", score=1.0),
            LabeledSpan(start=7, end=8, label="topic", score=1.0),
        ],
        "relations": [
            BinaryRelation(
                head=LabeledSpan(start=4, end=6, label="content", score=1.0),
                tail=LabeledSpan(start=7, end=8, label="topic", score=1.0),
                label="is_about",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=7, end=8, label="topic", score=1.0),
                tail=LabeledSpan(start=4, end=6, label="content", score=1.0),
                label="is_about",
                score=1.0,
            ),
        ],
    }


def test_decode_with_add_reversed_relations():
    doc = ExampleDocument(text="This is a dummy text about nothing. Trust me.")
    doc.entities.append(LabeledSpan(start=10, end=20, label="content"))
    doc.entities.append(LabeledSpan(start=27, end=34, label="topic"))
    doc.relations.append(
        BinaryRelation(head=doc.entities[0], tail=doc.entities[1], label="is_about")
    )

    taskmodule = PointerNetworkTaskModuleForEnd2EndRE(
        tokenizer_name_or_path="facebook/bart-base",
        relation_layer_name="relations",
        annotation_field_mapping={
            "entities": "labeled_spans",
            "relations": "binary_relations",
        },
        add_reversed_relations=True,
    )
    taskmodule.prepare(documents=[doc])

    task_encodings = taskmodule.encode(doc, encode_target=True)
    assert len(task_encodings) == 1
    decoded_annotations, statistics = taskmodule.decode_annotations(task_encodings[0].targets)
    assert decoded_annotations == {
        "entities": [
            LabeledSpan(start=4, end=6, label="content", score=1.0),
            LabeledSpan(start=7, end=8, label="topic", score=1.0),
        ],
        "relations": [
            BinaryRelation(
                head=LabeledSpan(start=4, end=6, label="content", score=1.0),
                tail=LabeledSpan(start=7, end=8, label="topic", score=1.0),
                label="is_about",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=7, end=8, label="topic", score=1.0),
                tail=LabeledSpan(start=4, end=6, label="content", score=1.0),
                label="is_about_reversed",
                score=1.0,
            ),
        ],
    }

    task_outputs = [task_encoding.targets for task_encoding in task_encodings]
    docs_with_predictions = taskmodule.decode(task_encodings, task_outputs)
    assert len(docs_with_predictions) == 1
    doc_with_predictions: ExampleDocument = docs_with_predictions[0]
    assert set(doc_with_predictions.entities.predictions) == set(doc_with_predictions.entities)
    assert set(doc_with_predictions.relations.predictions) == set(doc_with_predictions.relations)


@pytest.fixture()
def target_encoding(taskmodule, task_encoding_without_target):
    return taskmodule.encode_target(task_encoding_without_target)


def test_target_encoding(target_encoding, taskmodule):
    assert target_encoding is not None
    if taskmodule.partition_layer_name is None:
        assert asdict(target_encoding) == {
            "labels": [14, 14, 5, 11, 12, 3, 6, 17, 17, 4, 2, 2, 2, 2, 1],
            "constraints": [
                [0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
        }
    elif taskmodule.partition_layer_name == "sentences":
        assert asdict(target_encoding) == {
            "labels": [14, 14, 5, 11, 12, 3, 6, 1],
            "constraints": [
                [0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
        }
    else:
        raise Exception(f"unknown partition_layer_name: {taskmodule.partition_layer_name}")


def test_task_encoding_with_deduplicated_relations(caplog):
    doc = ExampleDocument(text="This is a dummy text about nothing. Trust me.")
    doc.entities.append(LabeledSpan(start=10, end=20, label="content"))
    doc.entities.append(LabeledSpan(start=27, end=34, label="topic"))
    doc.entities.append(LabeledSpan(start=42, end=44, label="person"))
    assert doc.entities.resolve() == [
        ("content", "dummy text"),
        ("topic", "nothing"),
        ("person", "me"),
    ]
    # add the same relation twice (just use a different score, but that should not matter)
    doc.relations.append(
        BinaryRelation(head=doc.entities[0], tail=doc.entities[1], label="is_about")
    )
    doc.relations.append(
        BinaryRelation(head=doc.entities[0], tail=doc.entities[1], label="is_about", score=0.9)
    )
    assert doc.relations.resolve() == [
        ("is_about", (("content", "dummy text"), ("topic", "nothing"))),
        ("is_about", (("content", "dummy text"), ("topic", "nothing"))),
    ]
    taskmodule = PointerNetworkTaskModuleForEnd2EndRE(
        tokenizer_name_or_path="facebook/bart-base",
        relation_layer_name="relations",
        annotation_field_mapping={
            "entities": "labeled_spans",
            "relations": "binary_relations",
        },
    )
    taskmodule.prepare(documents=[doc])
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        task_encodings = taskmodule.encode(doc, encode_target=True)
    messages = list(caplog.messages)

    assert len(task_encodings) == 1
    decoded_annotations, statistics = taskmodule.decode_annotations(task_encodings[0].targets)
    assert decoded_annotations == {
        "entities": [
            LabeledSpan(start=4, end=6, label="content", score=1.0),
            LabeledSpan(start=7, end=8, label="topic", score=1.0),
            LabeledSpan(start=10, end=11, label="person", score=1.0),
        ],
        "relations": [
            BinaryRelation(
                head=LabeledSpan(start=4, end=6, label="content", score=1.0),
                tail=LabeledSpan(start=7, end=8, label="topic", score=1.0),
                label="is_about",
                score=1.0,
            )
        ],
    }

    assert messages == [
        (
            "encoding errors: {'correct': 2}, skipped annotations:\n"
            "{\n"
            '  "relations": [\n'
            '    "BinaryRelation('
            "head=LabeledSpan(start=4, end=6, label='content', score=1.0), "
            "tail=LabeledSpan(start=7, end=8, label='topic', score=1.0), "
            "label='is_about', score=0.9"
            ')"\n'
            "  ]\n"
            "}"
        )
    ]


def test_task_encoding_with_conflicting_relations(caplog):
    doc = ExampleDocument(text="This is a dummy text about nothing. Trust me.")
    doc.entities.append(LabeledSpan(start=10, end=20, label="content"))
    doc.entities.append(LabeledSpan(start=27, end=34, label="topic"))
    doc.entities.append(LabeledSpan(start=42, end=44, label="person"))
    assert doc.entities.resolve() == [
        ("content", "dummy text"),
        ("topic", "nothing"),
        ("person", "me"),
    ]
    # add two relations with the same head and tail, but different labels
    doc.relations.append(
        BinaryRelation(head=doc.entities[0], tail=doc.entities[1], label="is_about")
    )
    doc.relations.append(
        BinaryRelation(head=doc.entities[0], tail=doc.entities[1], label="wrong_relation")
    )
    assert doc.relations.resolve() == [
        ("is_about", (("content", "dummy text"), ("topic", "nothing"))),
        ("wrong_relation", (("content", "dummy text"), ("topic", "nothing"))),
    ]
    taskmodule = PointerNetworkTaskModuleForEnd2EndRE(
        tokenizer_name_or_path="facebook/bart-base",
        relation_layer_name="relations",
        annotation_field_mapping={
            "entities": "labeled_spans",
            "relations": "binary_relations",
        },
    )
    taskmodule.prepare(documents=[doc])
    caplog.clear()
    with caplog.at_level(logging.ERROR):
        task_encodings = taskmodule.encode(doc, encode_target=True)
    messages = list(caplog.messages)

    assert len(task_encodings) == 0

    assert messages == [
        "failed to encode target, it will be skipped: "
        "relation ('Ġdummy', 'Ġtext') -> ('Ġnothing',) already exists, but has "
        "another label: is_about (current label: wrong_relation)."
    ]


@pytest.fixture()
def task_encoding(task_encoding_without_target, target_encoding):
    task_encoding_without_target.targets = target_encoding
    return task_encoding_without_target


def _separate_constraint(constraint, taskmodule):
    special_ids = sorted(taskmodule.special_target2id.values())
    none_ids = [taskmodule.none_id]
    span_ids = taskmodule.span_ids
    rel_ids = taskmodule.relation_ids
    result = [[constraint[id] for id in ids] for ids in [special_ids, none_ids, span_ids, rel_ids]]
    result += [constraint[taskmodule.pointer_offset :]]
    assert sum(len(con_part) for con_part in result) == len(constraint)
    return result


def test_build_constraint(taskmodule):
    target_ids = [14, 14, 5, 11, 12, 3, 6, 17, 17, 4, 2, 2, 2, 2, 1]
    input_len = 13

    # empty previous_ids
    constraint = taskmodule._build_constraint(previous_ids=[], input_len=input_len)
    # [bos, eos], [none], [content, person, topic], [is_about] [13 offsets (all remaining)]
    constraint_formatted = _separate_constraint(constraint.tolist(), taskmodule)
    # allow eos and all offsets
    assert constraint_formatted == [
        [0, 1],
        [0],
        [0, 0, 0],
        [0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]

    # just first span start
    constraint = taskmodule._build_constraint(previous_ids=[14], input_len=input_len)
    # [bos, eos], [none], [content, person, topic], [is_about] [13 offsets (all remaining)]
    constraint_formatted = _separate_constraint(constraint.tolist(), taskmodule)
    # allow all offsets after first span start
    assert constraint_formatted == [
        [0, 0],
        [0],
        [0, 0, 0],
        [0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    ]

    # first span start and end
    constraint = taskmodule._build_constraint(previous_ids=[14, 14], input_len=input_len)
    # [bos, eos], [none], [content, person, topic], [is_about] [13 offsets (all remaining)]
    constraint_formatted = _separate_constraint(constraint.tolist(), taskmodule)
    # allow all span ids
    assert constraint_formatted == [
        [0, 0],
        [0],
        [1, 1, 1],
        [0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    # first span start, end, and label
    constraint = taskmodule._build_constraint(previous_ids=[14, 14, 5], input_len=input_len)
    # [bos, eos], [none], [content, person, topic], [is_about] [13 offsets (all remaining)]
    constraint_formatted = _separate_constraint(constraint.tolist(), taskmodule)
    # allow none and all offsets except offsets covered by first span
    assert constraint_formatted == [
        [0, 0],
        [1],
        [0, 0, 0],
        [0],
        [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
    ]

    # first span, and second span start
    constraint = taskmodule._build_constraint(previous_ids=[14, 14, 5, 11], input_len=input_len)
    # [bos, eos], [none], [content, person, topic], [is_about] [13 offsets (all remaining)]
    constraint_formatted = _separate_constraint(constraint.tolist(), taskmodule)
    # allow all offsets after second span start, but not after first span start
    assert constraint_formatted == [
        [0, 0],
        [0],
        [0, 0, 0],
        [0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    ]

    # first span, and second span start and end
    constraint = taskmodule._build_constraint(
        previous_ids=[14, 14, 5, 11, 12], input_len=input_len
    )
    # [bos, eos], [none], [content, person, topic], [is_about] [13 offsets (all remaining)]
    constraint_formatted = _separate_constraint(constraint.tolist(), taskmodule)
    # allow all span ids
    assert constraint_formatted == [
        [0, 0],
        [0],
        [1, 1, 1],
        [0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    # first span, and second span
    constraint = taskmodule._build_constraint(
        previous_ids=[14, 14, 5, 11, 12, 3], input_len=input_len
    )
    # [bos, eos], [none], [content, person, topic], [is_about] [13 offsets (all remaining)]
    constraint_formatted = _separate_constraint(constraint.tolist(), taskmodule)
    # allow all relation ids
    assert constraint_formatted == [
        [0, 0],
        [0],
        [0, 0, 0],
        [1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    # fist span, and (1 to 3)-times none
    for i in range(1, 3):
        none_ids = [2] * i
        constraint = taskmodule._build_constraint(
            previous_ids=[14, 14, 5] + none_ids, input_len=input_len
        )
        # [bos, eos], [none], [content, person, topic], [is_about] [13 offsets (all remaining)]
        constraint_formatted = _separate_constraint(constraint.tolist(), taskmodule)
        # allow only none
        assert constraint_formatted == [
            [0, 0],
            [1],
            [0, 0, 0],
            [0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]

    # contains eos
    constraint = taskmodule._build_constraint(
        previous_ids=[14, 14, 5, 11, 12, 3, 6, 1], input_len=input_len
    )
    # [bos, eos/pad], [none], [content, person, topic], [is_about] [13 offsets (all remaining)]
    constraint_formatted = _separate_constraint(constraint.tolist(), taskmodule)
    # allow only pad (same as eos)
    assert constraint_formatted == [
        [0, 1],
        [0],
        [0, 0, 0],
        [0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]


def test_maybe_log_example(taskmodule, task_encoding, caplog, config):
    original_log_first_n_examples = taskmodule.log_first_n_examples
    taskmodule.log_first_n_examples = 1
    caplog.clear()
    with caplog.at_level(logging.INFO):
        taskmodule.maybe_log_example(task_encoding)
    if config == {}:
        assert caplog.messages == [
            "*** Example ***",
            "doc.id:       None-tokenized-1-of-1",
            "input_ids:    0 713 16 10 34759 2788 59 1085 4 3101 162 4 2",
            "input_tokens: <s> This Ġis Ġa Ġdummy Ġtext Ġabout Ġnothing . ĠTrust Ġme . " "</s>",
            "label_ids:    14 14 5 11 12 3 6 17 17 4 2 2 2 2 1",
            "label_tokens: 14 {Ġnothing} 14 {Ġnothing} topic 11 {Ġdummy} 12 {Ġtext} content is_about 17 {Ġme} 17 "
            "{Ġme} person none none none none </s>",
            "constraints:  torch.Size([15, 20]) (content is omitted)",
        ]
    elif config == {"partition_layer_name": "sentences"}:
        assert caplog.messages == [
            "*** Example ***",
            "doc.id:       None-tokenized-1-of-2",
            "input_ids:    0 713 16 10 34759 2788 59 1085 4 2",
            "input_tokens: <s> This Ġis Ġa Ġdummy Ġtext Ġabout Ġnothing . </s>",
            "label_ids:    14 14 5 11 12 3 6 1",
            "label_tokens: 14 {Ġnothing} 14 {Ġnothing} topic 11 {Ġdummy} 12 {Ġtext} content is_about </s>",
            "constraints:  torch.Size([8, 17]) (content is omitted)",
        ]
    else:
        raise Exception(f"unknown config: {config}")

    # restore original value
    taskmodule.log_first_n_examples = original_log_first_n_examples


def test_maybe_log_example_disabled(taskmodule, task_encoding, caplog):
    original_log_first_n_examples = taskmodule.log_first_n_examples
    taskmodule.log_first_n_examples = None
    caplog.clear()
    with caplog.at_level(logging.INFO):
        taskmodule.maybe_log_example(task_encoding)
    assert caplog.record_tuples == []

    # restore original value
    taskmodule.log_first_n_examples = original_log_first_n_examples


@pytest.fixture()
def task_encodings(taskmodule, document):
    return taskmodule.encode(documents=[document], encode_target=True)


@pytest.fixture()
def batch(taskmodule, task_encodings):
    return taskmodule.collate(task_encodings)


def test_collate(batch, taskmodule):
    inputs, targets = batch
    for tensor in inputs.values():
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.int64
    for tensor in targets.values():
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.int64
    inputs_lists = {k: inputs[k].tolist() for k in sorted(inputs)}
    targets_lists = {k: targets[k].tolist() for k in sorted(targets)}
    if taskmodule.partition_layer_name is None:
        assert inputs_lists == {
            "input_ids": [[0, 713, 16, 10, 34759, 2788, 59, 1085, 4, 3101, 162, 4, 2]],
            "attention_mask": [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
        }
        assert targets_lists == {
            "constraints": [
                [
                    [0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ],
            "labels": [[14, 14, 5, 11, 12, 3, 6, 17, 17, 4, 2, 2, 2, 2, 1]],
            "decoder_attention_mask": [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
        }
    elif taskmodule.partition_layer_name == "sentences":
        assert inputs_lists == {
            "input_ids": [
                [0, 713, 16, 10, 34759, 2788, 59, 1085, 4, 2],
                [0, 18823, 162, 4, 2, 1, 1, 1, 1, 1],
            ],
            "attention_mask": [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]],
        }
        assert targets_lists == {
            "constraints": [
                [
                    [0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                [
                    [0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, -1, -1, -1, -1, -1],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1],
                    [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, -1, -1, -1, -1, -1],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1],
                ],
            ],
            "labels": [[14, 14, 5, 11, 12, 3, 6, 1], [9, 9, 4, 2, 2, 2, 2, 1]],
            "decoder_attention_mask": [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ],
        }
    else:
        raise Exception(f"unknown partition_layer_name: {taskmodule.partition_layer_name}")


@pytest.fixture()
def unbatched_output(taskmodule, batch):
    inputs, targets = batch
    # because the model is trained to reproduce the target tokens, we can just use them as model prediction
    return taskmodule.unbatch_output(targets)


@pytest.fixture()
def task_outputs(unbatched_output):
    return unbatched_output


@pytest.fixture()
def task_output(task_outputs) -> LabelsAndOptionalConstraints:
    return task_outputs[0]


def test_task_output(task_output, taskmodule):
    output_list = task_output.labels
    if taskmodule.partition_layer_name is None:
        assert output_list == [14, 14, 5, 11, 12, 3, 6, 17, 17, 4, 2, 2, 2, 2, 1]
    elif taskmodule.partition_layer_name == "sentences":
        assert output_list == [14, 14, 5, 11, 12, 3, 6, 1]
    else:
        raise Exception(f"unknown partition_layer_name: {taskmodule.partition_layer_name}")


def _test_annotations_from_output(task_encodings, task_outputs, taskmodule, layer_names_expected):
    assert len(task_outputs) == len(task_encodings)

    # this needs to be outside the below loop because documents can contain duplicates
    # which would break the comparison when clearing predictions that were already added
    for task_encoding in task_encodings:
        for layer_name in layer_names_expected:
            task_encoding.document[layer_name].predictions.clear()

    layer_names: Set[str] = set()
    # Note: this list may contain duplicates!
    documents: List[Document] = []
    for i in range(len(task_outputs)):
        task_encoding = task_encodings[i]
        task_output = task_outputs[i]
        documents.append(task_encoding.document)

        for layer_name, annotation in taskmodule.create_annotations_from_output(
            task_encoding=task_encoding, task_output=task_output
        ):
            task_encoding.document[layer_name].predictions.append(annotation)
            layer_names.add(layer_name)

    assert layer_names == layer_names_expected

    for document in documents:
        for layer_name in layer_names:
            layer = {
                str(ann)
                for ann in document[layer_name].predictions
                if ann.label in taskmodule.labels_per_layer[layer_name]
            }
            layer_expected = {
                str(ann)
                for ann in document[layer_name]
                if ann.label in taskmodule.labels_per_layer[layer_name]
            }
            assert layer == layer_expected

    # this needs to be outside the above loop because documents can contain duplicates
    # which would break the comparison when clearing predictions too early
    for document in documents:
        for layer_name in layer_names:
            document[layer_name].predictions.clear()


def test_annotations_from_output(task_encodings, task_outputs, taskmodule):
    _test_annotations_from_output(
        taskmodule=taskmodule,
        task_encodings=task_encodings,
        task_outputs=task_outputs,
        layer_names_expected={"entities", "relations"},
    )


def get_default_taskmodule(**kwargs):
    taskmodule = PointerNetworkTaskModuleForEnd2EndRE(
        tokenizer_name_or_path="facebook/bart-base",
        labels_per_layer={
            "labeled_spans": ["content", "person", "topic"],
            "binary_relations": ["is_about"],
        },
        **kwargs,
    )
    taskmodule.post_prepare()
    return taskmodule


def test_configure_model_metric():
    taskmodule = get_default_taskmodule()
    metric = taskmodule.configure_model_metric()
    assert metric is not None
    values = metric.compute()
    assert values == {
        "binary_relations": {},
        "decoding_errors": {"all": 0.0},
        "exact_encoding_matches": 0.0,
        "labeled_spans": {},
    }

    model_output = {"labels": torch.tensor([[14, 14, 5, 11, 12, 3, 6, 17, 17, 4, 2, 2, 2, 2, 1]])}
    # test with expected == prediction
    metric.update(model_output, model_output)
    values = metric.compute()
    assert values == {
        "exact_encoding_matches": 1.0,
        "decoding_errors": {"correct": 1.0, "all": 0.0},
        "labeled_spans": {
            "content": {"recall": 1.0, "precision": 1.0, "f1": 1.0},
            "person": {"recall": 1.0, "precision": 1.0, "f1": 1.0},
            "topic": {"recall": 1.0, "precision": 1.0, "f1": 1.0},
            "macro": {"recall": 1.0, "precision": 1.0, "f1": 1.0},
            "micro": {"recall": 1.0, "precision": 1.0, "f1": 1.0},
        },
        "binary_relations": {
            "is_about": {"recall": 1.0, "precision": 1.0, "f1": 1.0},
            "macro": {"recall": 1.0, "precision": 1.0, "f1": 1.0},
            "micro": {"recall": 1.0, "precision": 1.0, "f1": 1.0},
        },
    }
    torch.random.manual_seed(42)
    # random_labels = torch.randint(0, 20, (1, 30))
    # split into random_labels1 and random_labels2 just for better code formatting
    random_labels1 = [0, 14, 4, 19, 2, 6, 18, 3, 0, 8, 8, 14, 2, 1]
    random_labels2 = [14, 6, 7, 8, 4, 1, 17, 9, 14, 7, 13, 15, 5, 12, 18, 13]
    labels_random = torch.tensor([random_labels1 + random_labels2])
    metric.reset()
    # test the case where we have mixed results (correct and wrong)
    metric.update(model_output, model_output)
    metric.update(prediction={"labels": labels_random}, expected=model_output)
    values = metric.compute()
    assert values == {
        "exact_encoding_matches": 0.5,
        "decoding_errors": {"correct": 0.5, "len": 0.25, "order": 0.25, "all": 0.5},
        "labeled_spans": {
            "person": {"recall": 0.5, "precision": 1.0, "f1": 0.6666666865348816},
            "topic": {"recall": 0.5, "precision": 1.0, "f1": 0.6666666865348816},
            "content": {"recall": 0.5, "precision": 1.0, "f1": 0.6666666865348816},
            "macro": {"recall": 0.5, "precision": 1.0, "f1": 0.6666666865348816},
            "micro": {"recall": 0.5, "precision": 1.0, "f1": 0.6666666865348816},
        },
        "binary_relations": {
            "is_about": {"recall": 0.5, "precision": 1.0, "f1": 0.6666666865348816},
            "macro": {"recall": 0.5, "precision": 1.0, "f1": 0.6666666865348816},
            "micro": {"recall": 0.5, "precision": 1.0, "f1": 0.6666666865348816},
        },
    }

    # ensure that the metric can be pickled
    pickle.dumps(metric)


def test_configure_model_generation():
    taskmodule = get_default_taskmodule()
    assert taskmodule.configure_model_generation() == {
        "no_repeat_ngram_size": 7,
    }


def test_configure_model_generation_with_constrained_generation():
    taskmodule = get_default_taskmodule(constrained_generation=True)
    generation_config = taskmodule.configure_model_generation()
    assert set(generation_config) == {"no_repeat_ngram_size", "logits_processor"}
    assert generation_config["no_repeat_ngram_size"] == 7
    logits_processor = generation_config["logits_processor"]
    assert isinstance(logits_processor, LogitsProcessorList)
    assert len(logits_processor) == 2
    assert isinstance(logits_processor[0], FinitizeLogitsProcessor)
    assert isinstance(logits_processor[1], PrefixConstrainedLogitsProcessorWithMaximum)


def test_prefix_allowed_tokens_fn_with_maximum():
    taskmodule = get_default_taskmodule()
    # not that this includes the leading bos token
    add_previous_input_ids = torch.tensor([0, 14, 14, 5, 11, 12, 3, 6, 17, 17, 4, 2, 2, 2, 2, 1])

    # empty input (first entry)
    allowed_ids = taskmodule._prefix_allowed_tokens_fn_with_maximum(
        batch_id=0, input_ids=add_previous_input_ids[:1], maximum=20
    )
    # allow the eos id [1] and all offset ids [7..19]
    assert allowed_ids == [1, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    # first span start
    allowed_ids = taskmodule._prefix_allowed_tokens_fn_with_maximum(
        batch_id=0, input_ids=add_previous_input_ids[:2], maximum=20
    )
    # allow all offset ids from first span start [14..19]
    assert allowed_ids == [14, 15, 16, 17, 18, 19]

    # first span start and end
    allowed_ids = taskmodule._prefix_allowed_tokens_fn_with_maximum(
        batch_id=0, input_ids=add_previous_input_ids[:3], maximum=20
    )
    # allow all span ids
    assert allowed_ids == [3, 4, 5]

    # first span start, end, and label
    allowed_ids = taskmodule._prefix_allowed_tokens_fn_with_maximum(
        batch_id=0, input_ids=add_previous_input_ids[:4], maximum=20
    )
    # allow none [2] and all offsets except offsets covered by first span [14]
    assert allowed_ids == [2, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19]

    # first span, and second span start
    allowed_ids = taskmodule._prefix_allowed_tokens_fn_with_maximum(
        batch_id=0, input_ids=add_previous_input_ids[:5], maximum=20
    )
    # allow all offsets from second span start [11], but before first span start [14] because it would be an overlap
    assert allowed_ids == [11, 12, 13]

    # first span, and second span start and end
    allowed_ids = taskmodule._prefix_allowed_tokens_fn_with_maximum(
        batch_id=0, input_ids=add_previous_input_ids[:6], maximum=20
    )
    # allow all span ids
    assert allowed_ids == [3, 4, 5]

    # first span, and second span
    allowed_ids = taskmodule._prefix_allowed_tokens_fn_with_maximum(
        batch_id=0, input_ids=add_previous_input_ids[:7], maximum=20
    )
    # allow all relation ids
    assert allowed_ids == [6]

    # entry begins (second entry)
    allowed_ids = taskmodule._prefix_allowed_tokens_fn_with_maximum(
        batch_id=0, input_ids=add_previous_input_ids[:8], maximum=20
    )
    # allow eos [1] and all offsets [7..19]
    assert allowed_ids == [1, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    # first span start
    allowed_ids = taskmodule._prefix_allowed_tokens_fn_with_maximum(
        batch_id=0, input_ids=add_previous_input_ids[:9], maximum=20
    )
    # allow all offsets from first span start [17..19]
    assert allowed_ids == [17, 18, 19]

    # first span start and end
    allowed_ids = taskmodule._prefix_allowed_tokens_fn_with_maximum(
        batch_id=0, input_ids=add_previous_input_ids[:10], maximum=20
    )
    # allow all span ids
    assert allowed_ids == [3, 4, 5]

    # first span start, end, and span label
    allowed_ids = taskmodule._prefix_allowed_tokens_fn_with_maximum(
        batch_id=0, input_ids=add_previous_input_ids[:11], maximum=20
    )
    # allow none [2] and all offsets except offsets covered by first span [17]
    assert allowed_ids == [2, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19]

    # first span, and none
    allowed_ids = taskmodule._prefix_allowed_tokens_fn_with_maximum(
        batch_id=0, input_ids=add_previous_input_ids[:12], maximum=20
    )
    # allow only none [2] because when the entry contains already a none id, it cannot be followed by anything else
    assert allowed_ids == [2]

    # first span, and none, and none
    allowed_ids = taskmodule._prefix_allowed_tokens_fn_with_maximum(
        batch_id=0, input_ids=add_previous_input_ids[:13], maximum=20
    )
    # allow only none [2] because when the entry contains already a none id, it cannot be followed by anything else
    assert allowed_ids == [2]

    # first span, and none, and none, and none
    allowed_ids = taskmodule._prefix_allowed_tokens_fn_with_maximum(
        batch_id=0, input_ids=add_previous_input_ids[:14], maximum=20
    )
    # allow only none [2] because when the entry contains already a none id, it cannot be followed by anything else
    assert allowed_ids == [2]

    # first span, and none, and none, and none, and none (second entry is complete)
    allowed_ids = taskmodule._prefix_allowed_tokens_fn_with_maximum(
        batch_id=0, input_ids=add_previous_input_ids[:15], maximum=20
    )
    # allow eos [1] and all offsets [7..19]
    assert allowed_ids == [1, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    # got an eos, so the sequence is complete
    allowed_ids = taskmodule._prefix_allowed_tokens_fn_with_maximum(
        batch_id=0, input_ids=add_previous_input_ids[:16], maximum=20
    )
    # allow only pad [1] (same as eos) because the sequence is complete
    assert allowed_ids == [1]
