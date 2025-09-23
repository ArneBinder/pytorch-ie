import pytest
import torch
from torch.nn import Embedding
from transformers import BartConfig
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.bart.modeling_bart import BartEncoder

from pytorch_ie.models.base_models import BartModelWithDecoderPositionIds
from pytorch_ie.models.base_models.bart_with_decoder_position_ids import (
    BartDecoderWithPositionIds,
    BartLearnedPositionalEmbeddingWithPositionIds,
)


def test_bart_learned_positional_embedding_with_position_ids():
    # Arrange
    torch.manual_seed(42)
    model = BartLearnedPositionalEmbeddingWithPositionIds(10, 6)
    input_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    position_ids_original = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    position_ids_different = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 2]])

    # Act
    original = model(input_ids=input_ids)
    replaced_original = model(input_ids=input_ids, position_ids=position_ids_original)
    replaced_different = model(input_ids=input_ids, position_ids=position_ids_different)

    # Assert
    assert original.shape == (1, 10, 6)
    assert replaced_original.shape == (1, 10, 6)
    torch.testing.assert_close(original, replaced_original)
    assert replaced_different.shape == (1, 10, 6)
    assert not torch.allclose(original, replaced_different)


@pytest.fixture(scope="module")
def bart_config():
    return BartConfig(
        vocab_size=30,
        d_model=10,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=20,
        decoder_ffn_dim=20,
        max_position_embeddings=10,
    )


@pytest.fixture(scope="module")
def bart_decoder_with_position_ids(bart_config):
    return BartDecoderWithPositionIds(config=bart_config)


def test_bart_decoder_with_position_ids(bart_decoder_with_position_ids):
    assert bart_decoder_with_position_ids is not None


def test_bart_decoder_with_position_ids_get_input_embeddings(bart_decoder_with_position_ids):
    input_embeddings = bart_decoder_with_position_ids.get_input_embeddings()
    assert input_embeddings is not None
    assert isinstance(input_embeddings, Embedding)
    assert input_embeddings.embedding_dim == 10
    assert input_embeddings.num_embeddings == 30


def test_bart_decoder_with_position_ids_set_input_embeddings(bart_decoder_with_position_ids):
    original_input_embeddings = bart_decoder_with_position_ids.get_input_embeddings()
    torch.manual_seed(42)
    new_input_embeddings = Embedding(
        original_input_embeddings.num_embeddings, original_input_embeddings.embedding_dim
    )
    bart_decoder_with_position_ids.set_input_embeddings(new_input_embeddings)
    input_embeddings = bart_decoder_with_position_ids.get_input_embeddings()
    assert input_embeddings == new_input_embeddings
    assert input_embeddings is not original_input_embeddings
    # recover original input embeddings
    bart_decoder_with_position_ids.set_input_embeddings(original_input_embeddings)


def test_bart_decoder_with_position_ids_forward(bart_decoder_with_position_ids):
    # Arrange
    model = bart_decoder_with_position_ids
    input_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
    position_ids_original = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
    position_ids_different = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2]])

    # Act
    torch.manual_seed(42)
    original = model(input_ids=input_ids)
    torch.manual_seed(42)
    replaced_original = model(input_ids=input_ids, position_ids=position_ids_original)
    torch.manual_seed(42)
    replaced_different = model(input_ids=input_ids, position_ids=position_ids_different)

    # Assert
    assert isinstance(original, BaseModelOutputWithPastAndCrossAttentions)
    assert original.last_hidden_state.shape == (1, 8, 10)
    assert isinstance(replaced_original, BaseModelOutputWithPastAndCrossAttentions)
    torch.testing.assert_close(original.last_hidden_state, replaced_original.last_hidden_state)

    assert isinstance(replaced_different, BaseModelOutputWithPastAndCrossAttentions)
    assert replaced_different.last_hidden_state.shape == (1, 8, 10)
    assert not torch.allclose(original.last_hidden_state, replaced_different.last_hidden_state)


def test_bart_decoder_with_position_ids_forward_with_inputs_embeds(bart_decoder_with_position_ids):
    # Arrange
    model = bart_decoder_with_position_ids
    inputs_embeds = torch.randn(1, 8, 10)
    position_ids_original = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
    position_ids_different = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2]])

    # Act
    torch.manual_seed(42)
    original = model(inputs_embeds=inputs_embeds)
    torch.manual_seed(42)
    replaced_original = model(inputs_embeds=inputs_embeds, position_ids=position_ids_original)
    torch.manual_seed(42)
    replaced_different = model(inputs_embeds=inputs_embeds, position_ids=position_ids_different)

    # Assert
    assert isinstance(original, BaseModelOutputWithPastAndCrossAttentions)
    assert original.last_hidden_state.shape == (1, 8, 10)
    assert isinstance(replaced_original, BaseModelOutputWithPastAndCrossAttentions)
    torch.testing.assert_close(original.last_hidden_state, replaced_original.last_hidden_state)

    assert isinstance(replaced_different, BaseModelOutputWithPastAndCrossAttentions)
    assert replaced_different.last_hidden_state.shape == (1, 8, 10)
    assert not torch.allclose(original.last_hidden_state, replaced_different.last_hidden_state)


def test_bart_decoder_with_position_ids_forward_wrong_position_ids_shape(
    bart_decoder_with_position_ids,
):
    # Arrange
    model = bart_decoder_with_position_ids
    input_ids = torch.tensor([[0, 1, 2, 3]])
    position_ids_wrong_shape = torch.tensor([[0, 1, 2]])

    # Act
    torch.manual_seed(42)
    with pytest.raises(ValueError) as excinfo:
        model(input_ids=input_ids, position_ids=position_ids_wrong_shape)
    assert (
        str(excinfo.value)
        == "Position IDs shape torch.Size([1, 3]) does not match input ids shape torch.Size([1, 4])."
    )


@pytest.fixture(scope="module")
def bart_model_with_decoder_position_ids(bart_config):
    torch.manual_seed(42)
    model = BartModelWithDecoderPositionIds(config=bart_config)
    model.train()
    return model


def test_bart_model_with_decoder_position_ids(bart_model_with_decoder_position_ids):
    assert bart_model_with_decoder_position_ids is not None


def test_bart_model_with_decoder_position_ids_get_input_embeddings(
    bart_model_with_decoder_position_ids,
):
    input_embeddings = bart_model_with_decoder_position_ids.get_input_embeddings()
    assert input_embeddings is not None
    assert isinstance(input_embeddings, Embedding)
    assert input_embeddings.embedding_dim == 10
    assert input_embeddings.num_embeddings == 30


def test_bart_model_with_decoder_position_ids_set_input_embeddings(
    bart_model_with_decoder_position_ids,
):
    original_input_embeddings = bart_model_with_decoder_position_ids.get_input_embeddings()
    torch.manual_seed(42)
    new_input_embeddings = Embedding(
        original_input_embeddings.num_embeddings, original_input_embeddings.embedding_dim
    )
    bart_model_with_decoder_position_ids.set_input_embeddings(new_input_embeddings)
    input_embeddings = bart_model_with_decoder_position_ids.get_input_embeddings()
    assert input_embeddings == new_input_embeddings
    assert input_embeddings is not original_input_embeddings
    # recover original input embeddings
    bart_model_with_decoder_position_ids.set_input_embeddings(original_input_embeddings)


def test_bart_model_with_decoder_position_ids_get_encoder(bart_model_with_decoder_position_ids):
    encoder = bart_model_with_decoder_position_ids.get_encoder()
    assert encoder is not None
    assert isinstance(encoder, BartEncoder)


def test_bart_model_with_decoder_position_ids_get_decoder(bart_model_with_decoder_position_ids):
    decoder = bart_model_with_decoder_position_ids.get_decoder()
    assert decoder is not None
    assert isinstance(decoder, BartDecoderWithPositionIds)


@pytest.mark.parametrize(
    "return_dict, prepare_encoder_outputs, output_everything",
    [(True, True, True), (False, False, False)],
)
def test_bart_model_with_decoder_position_forward(
    bart_model_with_decoder_position_ids, return_dict, prepare_encoder_outputs, output_everything
):
    model = bart_model_with_decoder_position_ids

    # Arrange
    model.eval()
    input_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
    position_ids_original = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
    position_ids_different = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2]])
    common_kwargs = {"input_ids": input_ids, "return_dict": return_dict}
    if prepare_encoder_outputs:
        common_kwargs["encoder_outputs"] = bart_model_with_decoder_position_ids.get_encoder()(
            input_ids=input_ids, return_dict=False
        )
    else:
        common_kwargs["encoder_outputs"] = None
    if output_everything:
        common_kwargs["output_attentions"] = True
        common_kwargs["output_hidden_states"] = True

    # Act
    original = model(**common_kwargs)[0]
    replaced_original = model(
        decoder_position_ids=position_ids_original,
        **common_kwargs,
    )[0]
    replaced_different = model(decoder_position_ids=position_ids_different, **common_kwargs)[0]

    # Assert
    assert isinstance(original, torch.FloatTensor)
    assert original.shape == (1, 8, 10)
    torch.testing.assert_close(
        original[0, :5, :3],
        torch.tensor(
            [
                [0.7589594721794128, 1.0452316999435425, 0.7063764333724976],
                [-0.12192550301551819, -0.9932114481925964, -0.722382664680481],
                [0.24711951613426208, -0.291597843170166, -1.0466505289077759],
                [1.1228691339492798, -0.0873560905456543, 1.534016728401184],
                [-1.1132177114486694, 0.2277398556470871, 1.6456809043884277],
            ]
        ),
    )
    torch.testing.assert_close(
        original.sum(dim=-1),
        torch.tensor(
            [
                [
                    0.0,
                    -1.1920928955078125e-07,
                    -1.1920928955078125e-07,
                    -2.682209014892578e-07,
                    5.960464477539063e-08,
                    5.960464477539063e-08,
                    2.384185791015625e-07,
                    -5.960464477539063e-08,
                ]
            ]
        ),
    )
    assert isinstance(replaced_original, torch.FloatTensor)
    torch.testing.assert_close(original, replaced_original)

    assert isinstance(replaced_different, torch.FloatTensor)
    assert replaced_different.shape == (1, 8, 10)
    torch.testing.assert_close(
        replaced_different[0, :5, :3],
        torch.tensor(
            [
                [0.7589594721794128, 1.0452316999435425, 0.7063764333724976],
                [-0.0127173513174057, -0.8127143383026123, -1.256797194480896],
                [1.0517312288284302, 0.037927787750959396, -0.28661563992500305],
                [0.5884698629379272, 0.9930593371391296, 1.3842554092407227],
                [0.6132885813713074, -1.0105736255645752, 2.361264228820801],
            ]
        ),
    )
    torch.testing.assert_close(
        replaced_different.sum(dim=-1),
        torch.tensor(
            [
                [
                    0.0,
                    -2.384185791015625e-07,
                    -1.7881393432617188e-07,
                    2.5331974029541016e-07,
                    1.4901161193847656e-07,
                    1.1920928955078125e-07,
                    -1.1920928955078125e-07,
                    -1.7881393432617188e-07,
                ]
            ]
        ),
    )
    assert not torch.allclose(replaced_different, original)
