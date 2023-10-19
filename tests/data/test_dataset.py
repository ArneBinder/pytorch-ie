from collections.abc import Iterator
from dataclasses import dataclass
from typing import Dict, Sequence, Union

import datasets
import numpy
import pytest
import torch

from pytorch_ie import Dataset, IterableDataset
from pytorch_ie.annotations import BinaryRelation, Label, LabeledSpan, Span
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.core.taskmodule import (
    IterableTaskEncodingDataset,
    TaskEncodingDataset,
    TaskEncodingSequence,
)
from pytorch_ie.data.dataset import get_pie_dataset_type
from pytorch_ie.documents import TextDocument
from pytorch_ie.taskmodules import TransformerSpanClassificationTaskModule
from tests import _HF_CONLL2003_IS_AVAILABLE, DATASET_BUILDERS_ROOT
from tests.conftest import TestDocument


@pytest.fixture(scope="module")
def taskmodule():
    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = TransformerSpanClassificationTaskModule(
        tokenizer_name_or_path=tokenizer_name_or_path,
        entity_annotation="entities",
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


def test_dataset(maybe_iterable_dataset):
    dataset = {
        k: list(v) if isinstance(v, IterableDataset) else v
        for k, v in maybe_iterable_dataset.items()
    }
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


def test_dataset_map(maybe_iterable_dataset):
    train_dataset = maybe_iterable_dataset["train"]

    def clear_relations(document):
        document.relations.clear()
        return document

    assert sum(len(doc.relations) for doc in train_dataset) == 7

    mapped_dataset1 = train_dataset.map(clear_relations)

    assert sum(len(doc.relations) for doc in mapped_dataset1) == 0
    assert sum(len(doc.relations) for doc in train_dataset) == 7


def test_dataset_map_batched(maybe_iterable_dataset):
    train_dataset = maybe_iterable_dataset["train"]

    def clear_relations_batched(documents):
        assert len(documents) == 2
        for document in documents:
            document.relations.clear()
        return documents

    assert sum(len(doc.relations) for doc in train_dataset) == 7

    mapped_dataset1 = train_dataset.map(clear_relations_batched, batched=True, batch_size=2)

    assert sum(len(doc.relations) for doc in mapped_dataset1) == 0
    assert sum(len(doc.relations) for doc in train_dataset) == 7


def test_dataset_map_with_result_document_type(maybe_iterable_dataset):
    @dataclass
    class TestDocument(TextDocument):
        sentences: AnnotationList[Span] = annotation_field(target="text")
        entities: AnnotationList[LabeledSpan] = annotation_field(target="text")
        relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")

    @dataclass
    class TestDocumentWithTokensButNoRelations(TextDocument):
        sentences: AnnotationList[Span] = annotation_field(target="text")
        tokens: AnnotationList[Span] = annotation_field(target="text")
        entities: AnnotationList[LabeledSpan] = annotation_field(target="text")

    def clear_relations_and_add_one_token(
        document: TestDocument,
    ) -> TestDocumentWithTokensButNoRelations:
        document.relations.clear()
        # the conversion here is not really necessary, but to have correct typing
        result = document.as_type(TestDocumentWithTokensButNoRelations)
        # subtract 1 to create a Span different from the sentence to account for
        # https://github.com/ChristophAlt/pytorch-ie/pull/222
        result.tokens.append(Span(0, len(document.text) - 1))
        return result

    train_dataset = maybe_iterable_dataset["train"]

    assert sum(len(doc.relations) for doc in train_dataset) == 7

    mapped_dataset1 = train_dataset.map(
        clear_relations_and_add_one_token,
        result_document_type=TestDocumentWithTokensButNoRelations,
    )

    assert sum(len(doc.relations) for doc in train_dataset) == 7

    doc0 = list(train_dataset)[0]
    doc0_mapped = list(mapped_dataset1)[0]
    assert len(doc0_mapped.tokens) == 1
    token = doc0_mapped.tokens[0]
    assert token.start == 0
    assert token.end == len(doc0.text) - 1
    # check field names because isinstance does not work (the code of the document types
    # is the same, but lives at different locations)
    assert {f.name for f in doc0.fields()} == {f.name for f in TestDocument.fields()}
    assert {f.name for f in doc0_mapped.fields()} == {
        f.name for f in TestDocumentWithTokensButNoRelations.fields()
    }


@pytest.mark.parametrize("encode_target", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("as_dataset", [False, True])
def test_dataset_with_taskmodule(
    maybe_iterable_dataset, taskmodule, model_output, encode_target, inplace, as_dataset
):
    train_dataset = maybe_iterable_dataset["train"]

    taskmodule.prepare(train_dataset)
    assert set(taskmodule.label_to_id.keys()) == {"PER", "ORG", "O"}
    assert [taskmodule.id_to_label[i] for i in range(3)] == ["O", "ORG", "PER"]
    assert taskmodule.label_to_id["O"] == 0

    as_task_encoding_sequence = not encode_target
    as_iterator = isinstance(train_dataset, (IterableDataset, Iterator))
    if as_task_encoding_sequence:
        if as_iterator:
            with pytest.raises(
                ValueError, match="can not return a TaskEncodingSequence as Iterator"
            ):
                taskmodule.encode(
                    train_dataset, encode_target=encode_target, as_dataset=as_dataset
                )
            return
        if as_dataset:
            with pytest.raises(
                ValueError, match="can not return a TaskEncodingSequence as a dataset"
            ):
                taskmodule.encode(
                    train_dataset, encode_target=encode_target, as_dataset=as_dataset
                )
            return

    task_encodings = taskmodule.encode(
        train_dataset, encode_target=encode_target, as_dataset=as_dataset
    )

    if as_iterator:
        if as_task_encoding_sequence:
            raise NotImplementedError("this is not yet implemented")
        if as_dataset:
            assert isinstance(task_encodings, IterableTaskEncodingDataset)
        else:
            assert isinstance(task_encodings, Iterator)
    else:
        if as_dataset:
            if as_task_encoding_sequence:
                raise NotImplementedError("this is not yet implemented")
            else:
                assert isinstance(task_encodings, TaskEncodingDataset)
        else:
            if as_task_encoding_sequence:
                assert isinstance(task_encodings, TaskEncodingSequence)
            else:
                assert isinstance(task_encodings, Sequence)

    task_encoding_list = list(task_encodings)
    assert len(task_encoding_list) == 8
    task_encoding = task_encoding_list[5]
    document = list(train_dataset)[5]
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

    if isinstance(train_dataset, Dataset):
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


@pytest.mark.skipif(
    not _HF_CONLL2003_IS_AVAILABLE,
    reason="the Huggingface conll2003 dataset is not reachable and the local PIE-variant depends on it",
)
def test_load_with_hf_datasets():
    dataset_path = DATASET_BUILDERS_ROOT / "conll2003"

    dataset = datasets.load_dataset(
        path=str(dataset_path),
    )

    assert set(dataset.keys()) == {"train", "validation", "test"}

    assert len(dataset["train"]) == 14041
    assert len(dataset["validation"]) == 3250
    assert len(dataset["test"]) == 3453


@pytest.mark.skipif(
    not _HF_CONLL2003_IS_AVAILABLE,
    reason="the Huggingface conll2003 dataset is not reachable and the remote PIE-variant depends on it",
)
def test_load_with_hf_datasets_from_hub():
    dataset = datasets.load_dataset(
        path="pie/conll2003",
    )

    assert set(dataset.keys()) == {"train", "validation", "test"}

    assert len(dataset["train"]) == 14041
    assert len(dataset["validation"]) == 3250
    assert len(dataset["test"]) == 3453


def test_get_pie_dataset_type(json_dataset, iterable_json_dataset):
    assert get_pie_dataset_type(json_dataset["train"]) == Dataset
    assert get_pie_dataset_type(iterable_json_dataset["train"]) == IterableDataset
    with pytest.raises(TypeError) as excinfo:
        get_pie_dataset_type("not a dataset")
    assert (
        str(excinfo.value)
        == "the dataset must be of type Dataset or IterableDataset, but is of type <class 'str'>"
    )


@dataclass
class TestDocumentWithLabel(TextDocument):
    label: AnnotationList[Label] = annotation_field()


def convert_to_document_with_label(document: TestDocument) -> TestDocumentWithLabel:
    result = TestDocumentWithLabel(text=document.text)
    result.label.append(Label(label="label"))
    return result


@pytest.fixture
def dataset_with_converter_functions(maybe_iterable_dataset) -> Union[Dataset, IterableDataset]:
    train_dataset: Union[Dataset, IterableDataset] = maybe_iterable_dataset["train"]
    assert len(train_dataset.document_converters) == 0

    train_dataset.register_document_converter(convert_to_document_with_label)
    return train_dataset


def test_register_document_converter_function(dataset_with_converter_functions):

    assert len(dataset_with_converter_functions.document_converters) == 1
    assert TestDocumentWithLabel in dataset_with_converter_functions.document_converters
    assert (
        dataset_with_converter_functions.document_converters[TestDocumentWithLabel]
        == convert_to_document_with_label
    )


@dataclass
class TestDocumentWithLabeledSpans(TextDocument):
    spans: AnnotationList[LabeledSpan] = annotation_field(target="text")


@pytest.fixture
def dataset_with_converter_mapping(maybe_iterable_dataset) -> Union[Dataset, IterableDataset]:
    train_dataset: Union[Dataset, IterableDataset] = maybe_iterable_dataset["train"]
    assert len(train_dataset.document_converters) == 0

    field_mapping = {"entities": "spans"}
    train_dataset.register_document_converter(
        converter=field_mapping, document_type=TestDocumentWithLabeledSpans
    )
    return train_dataset


def test_register_document_converter_mapping(dataset_with_converter_mapping):
    assert len(dataset_with_converter_mapping.document_converters) == 1
    assert TestDocumentWithLabeledSpans in dataset_with_converter_mapping.document_converters
    assert dataset_with_converter_mapping.document_converters[TestDocumentWithLabeledSpans] == {
        "entities": "spans"
    }


def test_to_document_type_function(dataset_with_converter_functions):
    assert dataset_with_converter_functions.document_type == TestDocument
    converted_dataset = dataset_with_converter_functions.to_document_type(TestDocumentWithLabel)
    assert converted_dataset.document_type == TestDocumentWithLabel

    assert len(converted_dataset.document_converters) == 0
    for doc in converted_dataset:
        assert isinstance(doc, TestDocumentWithLabel)
        assert len(doc.label) == 1
        assert doc.label[0].label == "label"


def test_to_document_type_mapping(dataset_with_converter_mapping):
    assert dataset_with_converter_mapping.document_type == TestDocument
    converted_dataset = dataset_with_converter_mapping.to_document_type(
        TestDocumentWithLabeledSpans
    )
    assert converted_dataset.document_type == TestDocumentWithLabeledSpans

    assert len(converted_dataset.document_converters) == 0
    for doc_converted, doc in zip(converted_dataset, dataset_with_converter_mapping):
        assert isinstance(doc, TestDocument)
        assert isinstance(doc_converted, TestDocumentWithLabeledSpans)
        assert "spans" in doc_converted
        assert doc_converted.spans == doc.entities
        original_annotation_field_names = {f.name for f in doc.annotation_fields()}
        assert original_annotation_field_names == {"sentences", "entities", "relations"}
        for annotation_field_name in original_annotation_field_names:
            assert annotation_field_name not in doc_converted


def test_to_document_type_noop(maybe_iterable_dataset):
    train_dataset: Union[Dataset, IterableDataset] = maybe_iterable_dataset["train"]
    assert len(train_dataset.document_converters) == 0
    train_dataset.register_document_converter(
        convert_to_document_with_label, document_type=TestDocument
    )
    assert train_dataset.document_type == TestDocument
    converted_dataset = train_dataset.to_document_type(TestDocument)
    # the conversion should be a noop
    assert converted_dataset.document_type == TestDocument
    assert converted_dataset == train_dataset
    assert len(converted_dataset.document_converters) == 1
    assert TestDocument in converted_dataset.document_converters
    assert converted_dataset.document_converters[TestDocument] == convert_to_document_with_label


def test_to_document_type_convert_and_cast(dataset_with_converter_functions):
    @dataclass
    class TestDocumentWithLabelAndSpans(TestDocumentWithLabel):
        label: AnnotationList[Label] = annotation_field()
        spans: AnnotationList[Span] = annotation_field(target="text")

    assert dataset_with_converter_functions.document_type == TestDocument
    # The only converter is registered for TestDocumentWithLabel, but we request a conversion to
    # TestDocumentWithLabelAndSpans which is a *subclass* of TestDocumentWithLabel. This is a valid type
    # and the conversion is performed by first converting to TestDocumentWithLabel and then casting
    # to TestDocumentWithLabelAndSpans.
    converted_dataset = dataset_with_converter_functions.to_document_type(
        TestDocumentWithLabelAndSpans
    )
    assert converted_dataset.document_type == TestDocumentWithLabelAndSpans

    assert len(converted_dataset.document_converters) == 0
    for converted_doc, doc in zip(converted_dataset, dataset_with_converter_functions):
        assert isinstance(doc, TestDocument)
        assert isinstance(converted_doc, TestDocumentWithLabelAndSpans)
        assert converted_doc.text == doc.text
        assert len(converted_doc.label) == 1
        assert converted_doc.label[0].label == "label"
        assert len(converted_doc.spans) == 0


def test_to_document_type_not_found(dataset_with_converter_functions):
    assert dataset_with_converter_functions.document_type == TestDocument

    @dataclass
    class TestDocumentWithSpans(TestDocument):
        spans: AnnotationList[Span] = annotation_field(target="text")

    # The only converter is registered for TestDocumentWithLabel, but we request a conversion to
    # TestDocumentWithSpans. This is not a valid type because it is neither a subclass nor a superclass of
    # TestDocumentWithLabel, so just a simple cast is performed.
    converted_dataset = dataset_with_converter_functions.to_document_type(TestDocumentWithSpans)
    assert converted_dataset.document_type == TestDocumentWithSpans

    assert len(converted_dataset.document_converters) == 0
    for converted_doc, doc in zip(converted_dataset, dataset_with_converter_functions):
        assert isinstance(doc, TestDocument)
        assert isinstance(converted_doc, TestDocumentWithSpans)
        assert converted_doc.text == doc.text
        annotation_field_names = {f.name for f in doc.annotation_fields()}
        assert annotation_field_names == {"sentences", "entities", "relations"}
        converted_annotation_filed_names = {f.name for f in converted_doc.annotation_fields()}
        assert converted_annotation_filed_names == {"sentences", "spans", "entities", "relations"}
        common_annotation_field_names = annotation_field_names & converted_annotation_filed_names
        for afn in common_annotation_field_names:
            assert converted_doc[afn] == doc[afn]
