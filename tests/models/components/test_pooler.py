import pytest
import torch

from pytorch_ie.models.components.pooler import (
    CLS_TOKEN,
    MENTION_POOLING,
    START_TOKENS,
    ArgumentWrappedPooler,
    AtIndexPooler,
    SpanMaxPooler,
    SpanMeanPooler,
    get_pooler_and_output_size,
    pool_cls,
)


@pytest.mark.parametrize(
    "pooler_type",
    [
        CLS_TOKEN,
        START_TOKENS,
        MENTION_POOLING,
    ],
)
def test_get_pooler_and_output_size(pooler_type):
    pooler, output_size = get_pooler_and_output_size(config={"type": pooler_type}, input_dim=20)
    assert pooler is not None
    if pooler_type == CLS_TOKEN:
        assert output_size == 20
    elif pooler_type in (START_TOKENS, MENTION_POOLING):
        # pre default, num_indices is 2
        assert output_size == 20 * 2
    else:
        raise ValueError(f"Unknown pooler type {pooler_type}")


@pytest.mark.parametrize("aggregate", ["max", "mean"])
def test_get_pooler_and_output_size_mention(aggregate):
    pooler, output_size = get_pooler_and_output_size(
        config={"type": MENTION_POOLING, "aggregate": aggregate}, input_dim=20
    )
    assert pooler is not None
    assert output_size == 20 * 2
    if aggregate == "max":
        assert isinstance(pooler, SpanMaxPooler)
    elif aggregate == "mean":
        assert isinstance(pooler, SpanMeanPooler)
    else:
        raise ValueError(f"Unknown aggregate type {aggregate}")


def test_get_pooler_and_output_size_mention_unknown_aggregate():
    with pytest.raises(ValueError) as excinfo:
        get_pooler_and_output_size(
            config={"type": MENTION_POOLING, "aggregate": "unknown"}, input_dim=20
        )
    assert str(excinfo.value) == 'Unknown aggregation method for mention pooling: "unknown"'


def test_get_pooler_and_output_size_wrong_type():
    with pytest.raises(ValueError) as excinfo:
        get_pooler_and_output_size(config={"type": "wrong_type"}, input_dim=20)
    assert str(excinfo.value) == 'Unknown pooler type "wrong_type"'


@pytest.fixture(scope="session")
def hidde_state():
    result = torch.tensor(
        [
            [[0.00, 0.01], [0.10, 0.11], [0.20, 0.21], [0.30, 0.31]],
            [[1.00, 1.01], [1.10, 1.11], [1.20, 1.21], [1.30, 1.31]],
        ]
    )
    # batch_size x sequence_length x hidden_size
    assert result.shape == (2, 4, 2)
    return result


def test_pool_cls(hidde_state):
    pooler = pool_cls
    output = pooler(hidden_state=hidde_state)
    assert output is not None
    batch_size = hidde_state.shape[0]
    hidden_size = hidde_state.shape[-1]
    assert output.shape == (batch_size, hidden_size)
    torch.testing.assert_close(output, hidde_state[:, 0, :])
    torch.testing.assert_close(output, torch.tensor([[0.00, 0.01], [1.00, 1.01]]))


def test_at_index_pooler(hidde_state):
    pooler = AtIndexPooler(input_dim=hidde_state.shape[-1], num_indices=2)
    indices = torch.tensor([[2, 0], [1, 0]])
    output = pooler(hidden_state=hidde_state, indices=indices)
    assert output is not None
    batch_size = hidde_state.shape[0]
    hidden_size = hidde_state.shape[-1]
    # times num_indices (=2) due to concat
    assert output.shape == (batch_size, hidden_size * 2)
    torch.testing.assert_close(
        output, torch.tensor([[0.20, 0.21, 0.00, 0.01], [1.10, 1.11, 1.00, 1.01]])
    )


def test_at_index_pooler_with_offset(hidde_state):
    # set the seed to make sure that we get the same missing embeddings
    torch.manual_seed(42)
    pooler = AtIndexPooler(input_dim=hidde_state.shape[-1], num_indices=2, offset=-1)
    indices = torch.tensor([[2, 1], [0, -10]])
    output = pooler(hidden_state=hidde_state, indices=indices)
    assert output is not None
    batch_size = hidde_state.shape[0]
    hidden_size = hidde_state.shape[-1]
    # times num_indices (=2) due to concat
    assert output.shape == (batch_size, hidden_size * 2)
    # the second batch element has out of bounds indices, so we expect the missing embeddings
    # it needs to be flattened, because the output is concatenated
    torch.testing.assert_close(output[1], pooler.missing_embeddings.view(-1))
    torch.testing.assert_close(
        output,
        torch.tensor(
            [
                [0.10, 0.11, 0.00, 0.01],
                [
                    0.33669036626815796,
                    0.12880940735340118,
                    0.23446236550807953,
                    0.23033303022384644,
                ],
            ]
        ),
    )


def test_at_index_pooler_wrong_indices_shapes(hidde_state):
    pooler = AtIndexPooler(input_dim=hidde_state.shape[-1], num_indices=2)
    indices = torch.tensor([[2, 0, 1], [1, 0, 0]])
    with pytest.raises(ValueError) as excinfo:
        pooler(hidden_state=hidde_state, indices=indices)
    assert str(excinfo.value) == "number of indices [3] has to be the same as num_types [2]"


def test_argument_wrapped_pooler(hidde_state):
    def dummy_pooler(hidden_state, y):
        return hidden_state[:, 0, :]

    pooler = ArgumentWrappedPooler(pooler=dummy_pooler, argument_mapping={"x": "y"})
    output = pooler(hidden_state=hidde_state, x="dummy")
    assert output is not None
    batch_size = hidde_state.shape[0]
    hidden_size = hidde_state.shape[-1]
    assert output.shape == (batch_size, hidden_size)
    torch.testing.assert_close(output, hidde_state[:, 0, :])


def test_span_max_pooler(hidde_state):
    pooler = SpanMaxPooler(input_dim=hidde_state.shape[-1], num_indices=2)
    start_indices = torch.tensor([[2, 0], [0, 1]])
    end_indices = torch.tensor([[3, 3], [1, 2]])
    output = pooler(hidden_state=hidde_state, start_indices=start_indices, end_indices=end_indices)
    assert output is not None
    batch_size = hidde_state.shape[0]
    hidden_size = hidde_state.shape[-1]
    # times num_indices (=2) due to concat
    assert output.shape == (batch_size, hidden_size * 2)
    torch.testing.assert_close(
        output, torch.tensor([[0.20, 0.21, 0.20, 0.21], [1.00, 1.01, 1.10, 1.11]])
    )


def test_span_max_pooler_wrong_start_indices_shape(hidde_state):
    pooler = SpanMaxPooler(input_dim=hidde_state.shape[-1], num_indices=2)
    start_indices = torch.tensor([[2, 0, 1], [0, 1, 0]])
    end_indices = torch.tensor([[3, 3], [1, 2]])
    with pytest.raises(ValueError) as excinfo:
        pooler(hidden_state=hidde_state, start_indices=start_indices, end_indices=end_indices)
    assert str(excinfo.value) == (
        "number of start indices [3] has to be the same as num_types [2]"
    )


def test_span_max_pooler_wrong_end_indices_shape(hidde_state):
    pooler = SpanMaxPooler(input_dim=hidde_state.shape[-1], num_indices=2)
    start_indices = torch.tensor([[2, 0], [0, 1]])
    end_indices = torch.tensor([[3, 3, 3], [1, 2, 1]])
    with pytest.raises(ValueError) as excinfo:
        pooler(hidden_state=hidde_state, start_indices=start_indices, end_indices=end_indices)
    assert str(excinfo.value) == ("number of end indices [3] has to be the same as num_types [2]")


def test_span_max_pooler_start_indices_bigger_than_end_indices(hidde_state):
    pooler = SpanMaxPooler(input_dim=hidde_state.shape[-1], num_indices=2)
    start_indices = torch.tensor([[2, 0], [0, 1]])
    end_indices = torch.tensor([[1, 3], [1, 2]])
    with pytest.raises(ValueError) as excinfo:
        pooler(hidden_state=hidde_state, start_indices=start_indices, end_indices=end_indices)
    assert str(excinfo.value) == (
        "values in start_indices have to be smaller than respective values in end_indices, but start_indices=\n"
        "tensor([[2, 0],\n"
        "        [0, 1]])\n "
        "and end_indices=\n"
        "tensor([[1, 3],\n"
        "        [1, 2]])"
    )


def test_span_mean_pooler(hidde_state):
    pooler = SpanMeanPooler(input_dim=hidde_state.shape[-1], num_indices=2)
    start_indices = torch.tensor([[2, 0], [0, 1]])
    end_indices = torch.tensor([[3, 3], [1, 2]])
    output = pooler(hidden_state=hidde_state, start_indices=start_indices, end_indices=end_indices)
    assert output is not None
    batch_size = hidde_state.shape[0]
    hidden_size = hidde_state.shape[-1]
    # times num_indices (=2) due to concat
    assert output.shape == (batch_size, hidden_size * 2)
    torch.testing.assert_close(
        output, torch.tensor([[0.20, 0.21, 0.10, 0.11], [1.00, 1.01, 1.10, 1.11]])
    )
