import pickle
from typing import Any, Dict, List, Sequence, Tuple

import pytest
import torch
from pie_core import Annotation, TaskEncoding
from pie_documents.annotations import AbstractiveSummary
from pie_documents.documents import (
    TextDocumentWithAbstractiveSummary,
    TokenDocumentWithAbstractiveSummary,
)

from pytorch_ie.models.common import VALIDATION
from pytorch_ie.taskmodules import TextToTextTaskModule
from pytorch_ie.taskmodules.text_to_text import (
    InputEncodingType,
    TargetEncodingType,
    TaskEncodingType,
    TaskOutputType,
)


@pytest.fixture(scope="module")
def documents():
    result = []

    doc = TextDocumentWithAbstractiveSummary(text="This is a test document")
    summary = AbstractiveSummary(text="a document")
    doc.abstractive_summary.append(summary)
    result.append(doc)

    doc = TextDocumentWithAbstractiveSummary(
        text="This is another test document which is a bit longer"
    )
    summary = AbstractiveSummary(text="a longer document")
    doc.abstractive_summary.append(summary)
    result.append(doc)

    return result


@pytest.fixture(scope="module")
def taskmodule():
    return TextToTextTaskModule(
        tokenizer_name_or_path="google/t5-efficient-tiny-nl2",
        document_type="pie_documents.documents.TextDocumentWithAbstractiveSummary",
        target_layer="abstractive_summary",
        target_annotation_type="pie_documents.annotations.AbstractiveSummary",
        tokenized_document_type="pie_documents.documents.TokenDocumentWithAbstractiveSummary",
        text_metric_type="torchmetrics.text.ROUGEScore",
    )


def test_taskmodule(taskmodule):
    assert taskmodule is not None
    assert taskmodule.document_type == TextDocumentWithAbstractiveSummary
    assert taskmodule.tokenized_document_type == TokenDocumentWithAbstractiveSummary
    assert taskmodule.target_annotation_type == AbstractiveSummary
    assert taskmodule.layer_names == ["abstractive_summary"]
    assert taskmodule.generation_config == {}


@pytest.fixture(scope="module")
def task_encodings(taskmodule, documents) -> Sequence[TaskEncodingType]:
    encodings = taskmodule.encode(documents, encode_target=True)
    assert all(isinstance(encoding, TaskEncoding) for encoding in encodings)
    assert len(encodings) == 2 == len(documents)
    assert encodings[0].document == documents[0]
    assert encodings[1].document == documents[1]
    return encodings


def test_maybe_log_example(taskmodule, task_encodings, caplog):
    counter_backup = taskmodule.log_first_n_examples

    taskmodule.log_first_n_examples = 1
    with caplog.at_level("INFO"):
        taskmodule.maybe_log_example(task_encodings[0])

    assert len(caplog.messages) == 3
    assert caplog.messages[0] == "input_ids: [100, 19, 3, 9, 794, 1708, 1]"
    assert caplog.messages[1] == "attention_mask: [1, 1, 1, 1, 1, 1, 1]"
    assert caplog.messages[2] == "labels: [3, 9, 1708, 1]"

    taskmodule.log_first_n_examples = counter_backup


@pytest.fixture(scope="module")
def input_encoding(taskmodule, task_encodings) -> InputEncodingType:
    assert len(task_encodings) > 0
    return task_encodings[0].inputs


def test_input_encoding(taskmodule, input_encoding):
    assert isinstance(input_encoding, InputEncodingType)
    assert input_encoding.input_ids == [100, 19, 3, 9, 794, 1708, 1]
    assert input_encoding.attention_mask == [1, 1, 1, 1, 1, 1, 1]

    tokens = taskmodule.tokenizer.convert_ids_to_tokens(input_encoding.input_ids)
    assert tokens == ["▁This", "▁is", "▁", "a", "▁test", "▁document", "</s>"]


@pytest.fixture(scope="module")
def metadata(taskmodule, task_encodings) -> Dict[str, Any]:
    assert len(task_encodings) > 0
    return task_encodings[0].metadata


def test_metadata(taskmodule, metadata):
    assert set(metadata) == {"tokenized_document", "guidance_annotation"}

    tokenized_document = metadata["tokenized_document"]
    assert isinstance(tokenized_document, TokenDocumentWithAbstractiveSummary)
    assert tokenized_document.tokens == ("▁This", "▁is", "▁", "a", "▁test", "▁document", "</s>")
    assert len(tokenized_document.abstractive_summary) == 1
    assert tokenized_document.abstractive_summary[0].text == "a document"


@pytest.fixture(scope="module")
def target_encoding(taskmodule, task_encodings) -> TargetEncodingType:
    assert len(task_encodings) > 0
    return task_encodings[0].targets


def test_target_encoding(taskmodule, target_encoding):
    assert isinstance(target_encoding, TargetEncodingType)
    assert target_encoding.labels == [3, 9, 1708, 1]
    assert target_encoding.decoder_attention_mask == [1, 1, 1, 1]


@pytest.fixture(scope="module")
def batch(taskmodule, task_encodings) -> List[TaskEncodingType]:
    result = taskmodule.collate(task_encodings)
    return result


def test_batch(taskmodule, batch):
    assert len(batch) == 2
    inputs, targets = batch

    assert set(inputs) == {"input_ids", "attention_mask"}
    torch.testing.assert_close(
        inputs["input_ids"],
        torch.tensor(
            [
                [100, 19, 3, 9, 794, 1708, 1, 0, 0, 0, 0, 0],
                [100, 19, 430, 794, 1708, 84, 19, 3, 9, 720, 1200, 1],
            ]
        ),
    )
    torch.testing.assert_close(
        inputs["attention_mask"],
        torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
    )

    assert set(targets) == {"labels", "decoder_attention_mask"}
    torch.testing.assert_close(
        targets["labels"], torch.tensor([[3, 9, 1708, 1, 0], [3, 9, 1200, 1708, 1]])
    )
    torch.testing.assert_close(
        targets["decoder_attention_mask"], torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
    )


@pytest.fixture(scope="module")
def unbatched_output(taskmodule, batch) -> Sequence[TaskOutputType]:
    inputs, targets = batch
    return taskmodule.unbatch_output(targets)


def test_unbatched_output(taskmodule, unbatched_output):
    assert all(isinstance(output, TargetEncodingType) for output in unbatched_output)
    assert len(unbatched_output) == 2

    assert unbatched_output[0].labels == [3, 9, 1708, 1]
    assert unbatched_output[0].decoder_attention_mask is None

    assert unbatched_output[1].labels == [3, 9, 1200, 1708, 1]
    assert unbatched_output[1].decoder_attention_mask is None


@pytest.fixture(scope="module")
def decoded_annotations(
    taskmodule, task_encodings, unbatched_output
) -> List[Tuple[str, Annotation]]:
    result = []
    for encoding, output in zip(task_encodings, unbatched_output):
        result.extend(
            taskmodule.create_annotations_from_output(task_encoding=encoding, task_output=output)
        )
    return result


def test_decoded_annotations(taskmodule, decoded_annotations):
    names, annotations = zip(*decoded_annotations)
    assert all(layer_name == taskmodule.target_layer for layer_name in names)
    assert all(
        isinstance(annotation, taskmodule.target_annotation_type) for annotation in annotations
    )

    assert len(annotations) == 2
    assert annotations[0].text == "a document"
    assert annotations[0].score is None
    assert annotations[1].text == "a longer document"
    assert annotations[1].score is None


def test_configure_model_metrics(taskmodule):
    metric = taskmodule.configure_model_metric(stage=VALIDATION)
    assert metric is not None
    values = metric.compute()
    keys = {
        "rouge2_fmeasure",
        "rougeL_recall",
        "rouge1_precision",
        "rouge1_recall",
        "rouge2_recall",
        "rougeL_precision",
        "rouge1_fmeasure",
        "rougeLsum_recall",
        "rougeLsum_precision",
        "rougeL_fmeasure",
        "rouge2_precision",
        "rougeLsum_fmeasure",
    }
    assert set(values) == keys
    assert all(torch.isnan(value) for value in values.values())

    labels = torch.tensor([[3, 9, 1708, 1, 0], [3, 9, 1200, 1708, 1]])
    metric.update(prediction={"labels": labels}, target={"labels": labels})
    assert set(metric.metric_state) == keys
    assert all(
        value == [torch.tensor(1.0), torch.tensor(1.0)] for value in metric.metric_state.values()
    )
    values = metric.compute()
    assert set(values) == keys
    assert all(value == torch.tensor(1.0) for value in values.values())

    random_labels = torch.tensor([[875, 885, 112, 289, 769], [270, 583, 970, 114, 71]])
    metric.update(prediction={"labels": random_labels}, target={"labels": labels})
    values = metric.compute()
    assert {k: v.item() for k, v in values.items()} == {
        "rouge1_fmeasure": 0.5625,
        "rouge1_precision": 0.550000011920929,
        "rouge1_recall": 0.5833333134651184,
        "rouge2_fmeasure": 0.5,
        "rouge2_precision": 0.5,
        "rouge2_recall": 0.5,
        "rougeL_fmeasure": 0.5625,
        "rougeL_precision": 0.550000011920929,
        "rougeL_recall": 0.5833333134651184,
        "rougeLsum_fmeasure": 0.5625,
        "rougeLsum_precision": 0.550000011920929,
        "rougeLsum_recall": 0.5833333134651184,
    }

    # ensure that the metric can be pickled
    pickle.dumps(metric)


def test_configure_model_generation(taskmodule):
    generation_config = taskmodule.configure_model_generation()
    assert generation_config is not None
    assert generation_config == {}


def test_warn_once(taskmodule, caplog):
    with caplog.at_level("WARNING"):
        taskmodule.warn_only_once("test")
        taskmodule.warn_only_once("test")
        taskmodule.warn_only_once("test2")

    assert len(caplog.messages) == 2
    assert caplog.messages[0] == "test (This warning will only be shown once)"
    assert caplog.messages[1] == "test2 (This warning will only be shown once)"
