import pytest
import torch
import transformers
from transformers import BatchEncoding
from transformers.modeling_outputs import Seq2SeqLMOutput

from pytorch_ie import TransformerSeq2SeqModel

LOSS = torch.rand(1).to(dtype=torch.float)
# a batch with one entry: 10 tokens from a 100-token vocabulary
TOKEN_IDS = torch.randint(0, 100, (1, 10))


class MockModel:
    def __init__(self) -> None:
        pass

    def __call__(self, *args, **kwargs):
        return Seq2SeqLMOutput(loss=LOSS)

    def generate(self, **kwargs):
        return TOKEN_IDS


@pytest.fixture
def mock_model(monkeypatch):
    monkeypatch.setattr(
        transformers.AutoModelForSeq2SeqLM,
        "from_config",
        lambda config: MockModel(),
    )
    monkeypatch.setattr(
        transformers.AutoModelForSeq2SeqLM,
        "from_pretrained",
        lambda pretrained_model_name_or_path: MockModel(),
    )

    model = TransformerSeq2SeqModel(model_name_or_path="some-model-name")
    assert not model.is_from_pretrained

    return model


@pytest.fixture(scope="module")
def batch():
    # taken from tests.taskmodules.test_transformer_seq2seq.test_collate()
    input_ids = torch.IntTensor(
        [
            [
                0,
                17,
                48,
                31845,
                10,
                2422,
                22307,
                11838,
                12,
                611,
                13552,
                5897,
                16,
                129,
                457,
                9,
                24,
                6,
                17,
                46,
                26,
                6002,
                2265,
                26942,
                6,
                937,
                1784,
                23,
                36363,
                846,
                8,
                4196,
                736,
                9,
                35890,
                40790,
                4,
                2,
            ],
        ]
    ).to(dtype=torch.int64)
    attention_mask = torch.IntTensor(
        [
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ],
        ]
    ).to(dtype=torch.int64)
    labels = torch.IntTensor(
        [
            [
                0,
                50267,
                11838,
                12,
                611,
                13552,
                5897,
                1437,
                50266,
                2422,
                1437,
                50265,
                34,
                1215,
                90,
                14631,
                2,
            ],
        ]
    ).to(dtype=torch.int64)

    encoding = BatchEncoding(
        data={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    )
    # IMPORTANT: We return a tuple (note the comma)!
    return (encoding,)


def test_forward(mock_model, batch):
    (inputs,) = batch
    output = mock_model.forward(inputs)

    assert isinstance(output, Seq2SeqLMOutput)
    loss = output.loss
    torch.testing.assert_close(loss, LOSS)


def test_predict(mock_model, batch):
    # taken from src.pipeline.Pipeline._forward
    inputs = batch[0]
    prediction = mock_model.predict(inputs=inputs.data)
    torch.testing.assert_close(prediction, TOKEN_IDS)


def test_training_step(mock_model, batch):
    loss = mock_model.training_step(batch, batch_idx=0)
    torch.testing.assert_close(loss, LOSS)


def test_validation_step(mock_model, batch):
    loss = mock_model.validation_step(batch, batch_idx=0)
    torch.testing.assert_close(loss, LOSS)


def test_test_step(mock_model, batch):
    loss = mock_model.test_step(batch, batch_idx=0)
    torch.testing.assert_close(loss, LOSS)
