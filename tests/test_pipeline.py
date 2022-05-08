import re

import pytest
import torch
import transformers
from transformers.modeling_outputs import BaseModelOutputWithPooling

import pytorch_ie.models.modules.mlp
from pytorch_ie.core.taskmodule import InplaceNotSupportedException
from pytorch_ie.models.transformer_span_classification import TransformerSpanClassificationModel
from pytorch_ie.pipeline import Pipeline
from pytorch_ie.taskmodules.transformer_span_classification import (
    TransformerSpanClassificationTaskModule,
)


@pytest.fixture(scope="module")
def taskmodule():
    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = TransformerSpanClassificationTaskModule(
        tokenizer_name_or_path=tokenizer_name_or_path
    )
    return taskmodule


@pytest.fixture
def prepared_taskmodule(taskmodule, documents):
    taskmodule.prepare(documents)
    return taskmodule


class MockConfig:
    def __init__(self, hidden_size: int = 10, classifier_dropout: float = 1.0) -> None:
        self.hidden_size = hidden_size
        self.classifier_dropout = classifier_dropout


class MockModel:
    def __init__(self, batch_size, seq_len, hidden_size) -> None:
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size

    def __call__(self, *args, **kwargs):
        last_hidden_state = torch.rand(self.batch_size, self.seq_len, self.hidden_size)
        return BaseModelOutputWithPooling(last_hidden_state=last_hidden_state)


@pytest.fixture
def mock_model(monkeypatch, documents, prepared_taskmodule):
    documents = documents[:3]

    encodings = prepared_taskmodule.encode(documents, encode_target=True)

    inputs, _ = prepared_taskmodule.collate(encodings)

    batch_size, seq_len = inputs["input_ids"].shape
    hidden_size = 10
    num_classes = 3

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda model_name_or_path: MockConfig(hidden_size=hidden_size, classifier_dropout=1.0),
    )
    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        lambda model_name_or_path, config: MockModel(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        ),
    )
    monkeypatch.setattr(
        pytorch_ie.models.modules.mlp.MLP,
        "__call__",
        lambda s, x: torch.tensor([0.0, 1.0, 0.0]).reshape(1, -1).expand(x.shape[0], -1),
    )

    return TransformerSpanClassificationModel(
        model_name_or_path="some-model-name",
        num_classes=num_classes,
        t_total=1,
        span_length_embedding_dim=15,
        max_span_length=2,
    )


@pytest.mark.slow
@pytest.mark.parametrize("inplace", [False, True])
def test_pipeline_with_document(documents, prepared_taskmodule, mock_model, inplace):
    document = documents[1]
    pipeline = Pipeline(model=mock_model, taskmodule=prepared_taskmodule, device=-1)

    returned_document = pipeline(document, inplace=inplace)

    if inplace:
        assert id(returned_document) == id(document)
        assert document.entities.predictions
        assert returned_document.entities.predictions
    else:
        assert not (id(returned_document) == id(document))
        assert not document.entities.predictions
        assert returned_document.entities.predictions


@pytest.mark.slow
@pytest.mark.parametrize("inplace", [False, True])
def test_pipeline_with_documents(documents, prepared_taskmodule, mock_model, inplace):
    pipeline = Pipeline(model=mock_model, taskmodule=prepared_taskmodule, device=-1)

    returned_documents = pipeline(documents, inplace=inplace)

    assert len(documents) == len(returned_documents)

    for returned_document, document in zip(returned_documents, documents):
        if inplace:
            assert id(returned_document) == id(document)
            assert document.entities.predictions
            assert returned_document.entities.predictions
        else:
            assert not (id(returned_document) == id(document))
            assert not document.entities.predictions
            assert returned_document.entities.predictions


@pytest.mark.slow
@pytest.mark.parametrize("inplace", [False, True])
def test_pipeline_with_dataset(dataset, prepared_taskmodule, mock_model, inplace):
    train_dataset = dataset["train"]

    pipeline = Pipeline(model=mock_model, taskmodule=prepared_taskmodule, device=-1)

    if inplace:
        with pytest.raises(
            InplaceNotSupportedException,
            match=re.escape("Datasets can't be modified in place. Please set inplace=False."),
        ):
            returned_documents = pipeline(train_dataset, inplace=inplace)
    else:
        returned_documents = pipeline(train_dataset, inplace=inplace)

        assert len(train_dataset) == len(returned_documents)

        for returned_document, document in zip(returned_documents, train_dataset):
            assert not (id(returned_document) == id(document))
            assert not document.entities.predictions
            assert returned_document.entities.predictions


@pytest.mark.slow
def test_pipeline_with_dataset_never_cached(dataset, prepared_taskmodule, mock_model):

    train_dataset = dataset["train"]

    pipeline = Pipeline(model=mock_model, taskmodule=prepared_taskmodule, device=-1, inplace=False)

    returned_documents1 = pipeline(train_dataset, predict_field="entities")
    returned_documents2 = pipeline(train_dataset, predict_field="entities")

    assert returned_documents1._fingerprint != returned_documents2._fingerprint
