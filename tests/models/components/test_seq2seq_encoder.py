import pytest
import torch

from pytorch_ie.models.components.seq2seq_encoder import (
    ACTIVATION_TYPE2CLASS,
    RNN_TYPE2CLASS,
    build_seq2seq_encoder,
)


def test_no_encoder():
    seq2seq_dict = {}
    input_size = 10
    encoder, output_size = build_seq2seq_encoder(seq2seq_dict, input_size)
    assert encoder is None
    assert output_size == input_size

    seq2seq_dict = {
        "type": "sequential",
        "rnn_layer": {
            "type": "none",
        },
    }
    input_size = 10
    encoder, output_size = build_seq2seq_encoder(seq2seq_dict, input_size)
    assert len(encoder) == 0
    assert output_size == input_size


@pytest.mark.parametrize("seq2seq_enc_type", list(RNN_TYPE2CLASS))
@pytest.mark.parametrize("bidirectional", [True, False])
def test_rnn_encoder(seq2seq_enc_type, bidirectional):
    hidden_size = 99
    seq2seq_dict = {
        "type": seq2seq_enc_type,
        "hidden_size": hidden_size,
        "bidirectional": bidirectional,
    }
    input_size = 10
    encoder, output_size = build_seq2seq_encoder(seq2seq_dict, input_size)
    assert encoder is not None
    assert isinstance(encoder.rnn, RNN_TYPE2CLASS[seq2seq_enc_type])

    expected_output_size = hidden_size * 2 if bidirectional else hidden_size
    assert output_size is not None
    assert output_size == expected_output_size


@pytest.mark.parametrize("activation_type", list(ACTIVATION_TYPE2CLASS))
def test_activations(activation_type):
    seq2seq_dict = {
        "type": activation_type,
    }
    input_size = 10
    encoder, output_size = build_seq2seq_encoder(seq2seq_dict, input_size)
    assert encoder is not None
    assert isinstance(encoder, ACTIVATION_TYPE2CLASS[activation_type])
    assert output_size == input_size


def test_dropout():
    seq2seq_dict = {
        "type": "dropout",
        "p": 0.5,
    }
    input_size = 10
    encoder, output_size = build_seq2seq_encoder(seq2seq_dict, input_size)
    assert encoder is not None
    assert isinstance(encoder, torch.nn.Dropout)
    assert output_size == input_size


def test_linear():
    out_features = 99
    seq2seq_dict = {
        "type": "linear",
        "out_features": out_features,
    }

    input_size = 10
    encoder, output_size = build_seq2seq_encoder(seq2seq_dict, input_size)
    assert encoder is not None
    assert isinstance(encoder, torch.nn.Linear)
    assert output_size == out_features


def test_unknown_rnn_type():
    seq2seq_dict = {
        "type": "unknown",
    }
    with pytest.raises(ValueError) as exc_info:
        build_seq2seq_encoder(seq2seq_dict, 10)
    assert str(exc_info.value) == "Unknown seq2seq_encoder_type: unknown"
