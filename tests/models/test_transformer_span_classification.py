import pytest
import torch
import transformers
from transformers.modeling_outputs import BaseModelOutputWithPooling

from pytorch_ie.models.transformer_span_classification import TransformerSpanClassificationModel
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
    num_classes = 5

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

    model = TransformerSpanClassificationModel(
        model_name_or_path="some-model-name",
        num_classes=num_classes,
        span_length_embedding_dim=15,
        max_span_length=2,
        # disable warmup because it would require a trainer and a datamodule to get the total number of training steps
        warmup_proportion=0.0,
    )
    assert not model.is_from_pretrained

    return model


@pytest.mark.parametrize("no_attention_mask", [False, True])
def test_forward(documents, prepared_taskmodule, mock_model, no_attention_mask):
    documents = documents[:3]

    encodings = prepared_taskmodule.encode(documents, encode_target=False)

    inputs, _ = prepared_taskmodule.collate(encodings)

    if no_attention_mask:
        inputs.pop("attention_mask")

    num_classes = 5
    num_spans = 57 if no_attention_mask else 47

    output = mock_model(inputs)

    assert set(output.keys()) == {"logits", "batch_indices", "start_indices", "end_indices"}
    assert output["logits"].shape == (num_spans, num_classes)
    assert all(
        [
            output[key].shape == (num_spans,)
            for key in ["batch_indices", "start_indices", "end_indices"]
        ]
    )


@pytest.mark.parametrize("no_attention_mask", [False, True])
def test_training_step(documents, prepared_taskmodule, mock_model, no_attention_mask):
    documents = documents[:3]

    encodings = prepared_taskmodule.encode(documents, encode_target=True)

    inputs, targets = prepared_taskmodule.collate(encodings)

    if no_attention_mask:
        inputs.pop("attention_mask")

    loss = mock_model.training_step((inputs, targets), batch_idx=0)

    assert len(loss.shape) == 0


@pytest.mark.parametrize("no_attention_mask", [False, True])
def test_validation_step(documents, prepared_taskmodule, mock_model, no_attention_mask):
    documents = documents[:3]

    encodings = prepared_taskmodule.encode(documents, encode_target=True)

    inputs, targets = prepared_taskmodule.collate(encodings)

    if no_attention_mask:
        inputs.pop("attention_mask")

    loss = mock_model.validation_step((inputs, targets), batch_idx=0)

    assert len(loss.shape) == 0


def test_configure_optimizers(mock_model):
    optimizer = mock_model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Optimizer)


@pytest.mark.parametrize("seq_lengths", [None, [3, 4]])
def test_start_end_and_span_length_span_index(mock_model, seq_lengths):
    (
        start_indices,
        end_indices,
        span_length,
        batch_indices,
        offsets,
    ) = mock_model._start_end_and_span_length_span_index(
        batch_size=2, max_seq_length=4, seq_lengths=seq_lengths
    )

    if seq_lengths is None:
        assert torch.equal(start_indices, torch.tensor([0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2]))
        assert torch.equal(end_indices, torch.tensor([0, 1, 2, 3, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3]))
        assert torch.equal(span_length, torch.tensor([0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1]))
        assert torch.equal(batch_indices, torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]))
        assert torch.equal(offsets, torch.tensor([0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4]))
    else:
        assert torch.equal(start_indices, torch.tensor([0, 1, 2, 0, 1, 0, 1, 2, 3, 0, 1, 2]))
        assert torch.equal(end_indices, torch.tensor([0, 1, 2, 1, 2, 0, 1, 2, 3, 1, 2, 3]))
        assert torch.equal(span_length, torch.tensor([0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1]))
        assert torch.equal(batch_indices, torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]))
        assert torch.equal(offsets, torch.tensor([0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4]))
