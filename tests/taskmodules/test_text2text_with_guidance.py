from typing import Any, Dict, List, Sequence, Tuple

import pytest
import torch
from pie_core import Annotation, TaskEncoding
from pie_documents.annotations import GenerativeAnswer, Question
from pie_documents.documents import (
    TextDocumentWithQuestionsAndGenerativeAnswers,
    TokenDocumentWithQuestionsAndGenerativeAnswers,
)

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

    doc = TextDocumentWithQuestionsAndGenerativeAnswers(text="This is a test document")
    question = Question(text="What is this?")
    doc.questions.append(question)
    answer = GenerativeAnswer(text="a document", question=question)
    doc.generative_answers.append(answer)
    result.append(doc)

    doc = TextDocumentWithQuestionsAndGenerativeAnswers(
        text="This is another test document which is a bit longer."
    )
    question = Question(text="And what is this?")
    doc.questions.append(question)
    answer = GenerativeAnswer(text="a longer document", question=question)
    doc.generative_answers.append(answer)
    result.append(doc)

    return result


@pytest.fixture(scope="module")
def taskmodule():
    return TextToTextTaskModule(
        tokenizer_name_or_path="google/t5-efficient-tiny-nl2",
        document_type="pie_documents.documents.TextDocumentWithQuestionsAndGenerativeAnswers",
        target_layer="generative_answers",
        target_annotation_type="pie_documents.annotations.GenerativeAnswer",
        tokenized_document_type="pie_documents.documents.TokenDocumentWithQuestionsAndGenerativeAnswers",
        guidance_layer="questions",
        guidance_annotation_field="question",
        text_metric_type="torchmetrics.text.ROUGEScore",
    )


def test_taskmodule(taskmodule):
    assert taskmodule is not None
    assert taskmodule.document_type == TextDocumentWithQuestionsAndGenerativeAnswers
    assert taskmodule.tokenized_document_type == TokenDocumentWithQuestionsAndGenerativeAnswers
    assert taskmodule.target_annotation_type == GenerativeAnswer
    assert taskmodule.layer_names == ["generative_answers"]
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
    assert caplog.messages[0] == "input_ids: [363, 19, 48, 58, 1, 100, 19, 3, 9, 794, 1708, 1]"
    assert caplog.messages[1] == "attention_mask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]"
    assert caplog.messages[2] == "labels: [3, 9, 1708, 1]"

    taskmodule.log_first_n_examples = counter_backup


@pytest.fixture(scope="module")
def input_encoding(taskmodule, task_encodings) -> InputEncodingType:
    assert len(task_encodings) > 0
    return task_encodings[0].inputs


def test_input_encoding(taskmodule, input_encoding):
    assert isinstance(input_encoding, InputEncodingType)
    assert input_encoding.input_ids == [363, 19, 48, 58, 1, 100, 19, 3, 9, 794, 1708, 1]
    assert input_encoding.attention_mask == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    tokens = taskmodule.tokenizer.convert_ids_to_tokens(input_encoding.input_ids)
    assert tokens == [
        "▁What",
        "▁is",
        "▁this",
        "?",
        "</s>",
        "▁This",
        "▁is",
        "▁",
        "a",
        "▁test",
        "▁document",
        "</s>",
    ]


@pytest.fixture(scope="module")
def metadata(taskmodule, task_encodings) -> Dict[str, Any]:
    assert len(task_encodings) > 0
    return task_encodings[0].metadata


def test_metadata(taskmodule, metadata):
    assert set(metadata) == {"tokenized_document", "guidance_annotation"}

    tokenized_document = metadata["tokenized_document"]
    assert isinstance(tokenized_document, TokenDocumentWithQuestionsAndGenerativeAnswers)
    assert tokenized_document.tokens == (
        "▁What",
        "▁is",
        "▁this",
        "?",
        "</s>",
        "▁This",
        "▁is",
        "▁",
        "a",
        "▁test",
        "▁document",
        "</s>",
    )
    assert len(tokenized_document.questions) == 1
    assert tokenized_document.questions[0].text == "What is this?"


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
                [363, 19, 48, 58, 1, 100, 19, 3, 9, 794, 1708, 1, 0, 0, 0, 0, 0, 0, 0],
                [275, 125, 19, 48, 58, 1, 100, 19, 430, 794, 1708, 84, 19, 3, 9, 720, 1200, 5, 1],
            ]
        ),
    )
    torch.testing.assert_close(
        inputs["attention_mask"],
        torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        ),
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
