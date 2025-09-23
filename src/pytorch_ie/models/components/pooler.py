import logging
from typing import Any, Callable, Dict, Tuple, Union

import torch
from torch import Tensor, cat, nn

# possible pooler types
CLS_TOKEN = "cls_token"  # CLS token
START_TOKENS = "start_tokens"  # MTB start tokens concat
MENTION_POOLING = "mention_pooling"  # mention token pooling and concat


logger = logging.getLogger(__name__)


def pool_cls(hidden_state: Tensor, **kwargs) -> Tensor:
    return hidden_state[:, 0, :]


class AtIndexPooler(nn.Module):
    """Pooler that takes the hidden states at given indices. If the index is negative, a learned
    embedding is used.

    The indices are expected to have the shape [batch_size, num_indices]. The resulting embeddings are concatenated,
    so the output shape is [batch_size, num_indices * input_dim].

    Args:
        input_dim: The input dimension of the hidden state.
        num_indices: The number of indices to pool.
        offset: An offset to add to the indices. This can be useful if the input is prepared with special
            tokens at the beginning / at the end of indexed sequences, and we want to use the hidden state of this
            token instead of the first / last token of the sequence.

    Returns:
        The pooled hidden states with shape [batch_size, num_indices * input_dim].
    """

    def __init__(self, input_dim: int, num_indices: int = 2, offset: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.num_indices = num_indices
        self.offset = offset
        self.missing_embeddings = nn.Parameter(torch.empty(num_indices, self.input_dim))
        nn.init.normal_(self.missing_embeddings)

    def forward(self, hidden_state: Tensor, indices: Tensor, **kwargs) -> Tensor:
        batch_size, seq_len, hidden_size = hidden_state.shape
        if indices.shape[1] != self.num_indices:
            raise ValueError(
                f"number of indices [{indices.shape[1]}] has to be the same as num_types [{self.num_indices}]"
            )

        # respect the offset
        indices = indices + self.offset

        # times num_types due to concat
        result = torch.zeros(
            batch_size, hidden_size * self.num_indices, device=hidden_state.device
        )
        for batch_idx, current_indices in enumerate(indices):
            current_embeddings = [
                (
                    hidden_state[batch_idx, current_indices[i], :]
                    if current_indices[i] >= 0
                    else self.missing_embeddings[i]
                )
                for i in range(self.num_indices)
            ]
            result[batch_idx] = cat(current_embeddings, 0)
        return result

    @property
    def output_dim(self) -> int:
        return self.input_dim * self.num_indices


class ArgumentWrappedPooler(nn.Module):
    """Wraps a pooler and maps the arguments to the pooler.

    Args:
        pooler: The pooler to wrap.
        argument_mapping: A mapping from the arguments of the forward method to the arguments of the pooler.
    """

    def __init__(
        self, pooler: Union[nn.Module, Callable], argument_mapping: Dict[str, str], **kwargs
    ):
        super().__init__(**kwargs)
        self.pooler = pooler
        self.argument_mapping = argument_mapping

    def forward(self, hidden_state: Tensor, **kwargs) -> Tensor:
        pooler_kwargs = {}
        for k, v in kwargs.items():
            if k in self.argument_mapping:
                pooler_kwargs[self.argument_mapping[k]] = v
        return self.pooler(hidden_state, **pooler_kwargs)


class SpanMaxPooler(nn.Module):
    """Pooler that takes the max hidden state over spans. If the start or end index is negative, a
    learned.

    embedding is used. The indices are expected to have the shape [batch_size, num_indices]. The resulting embeddings
    are concatenated, so the output shape is [batch_size, num_indices * input_dim].

    Args:
        input_dim: The input dimension of the hidden state.
        num_indices: The number of indices to pool.

    Returns:
        The pooled hidden states with shape [batch_size, num_indices * input_dim].
    """

    def __init__(self, input_dim: int, num_indices: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.num_indices = num_indices
        self.missing_embeddings = nn.Parameter(torch.empty(num_indices, self.input_dim))
        nn.init.normal_(self.missing_embeddings)

    def forward(
        self, hidden_state: Tensor, start_indices: Tensor, end_indices: Tensor, **kwargs
    ) -> Tensor:
        batch_size, seq_len, hidden_size = hidden_state.shape
        if start_indices.shape[1] != self.num_indices:
            raise ValueError(
                f"number of start indices [{start_indices.shape[1]}] has to be the same as num_types [{self.num_indices}]"
            )

        if end_indices.shape[1] != self.num_indices:
            raise ValueError(
                f"number of end indices [{end_indices.shape[1]}] has to be the same as num_types [{self.num_indices}]"
            )

        # check that start_indices are before end_indices
        mask_both_positive = (start_indices >= 0) & (end_indices >= 0)
        mask_start_before_end = start_indices < end_indices
        mask_valid = mask_start_before_end | ~mask_both_positive
        if not torch.all(mask_valid):
            raise ValueError(
                f"values in start_indices have to be smaller than respective values in "
                f"end_indices, but start_indices=\n{start_indices}\n and end_indices=\n{end_indices}"
            )

        # times num_indices due to concat
        result = torch.zeros(
            batch_size, hidden_size * self.num_indices, device=hidden_state.device
        )
        for batch_idx in range(batch_size):
            current_start_indices = start_indices[batch_idx]
            current_end_indices = end_indices[batch_idx]
            current_embeddings = [
                (
                    torch.amax(
                        hidden_state[
                            batch_idx, current_start_indices[i] : current_end_indices[i], :
                        ],
                        0,
                    )
                    if current_start_indices[i] >= 0 and current_end_indices[i] >= 0
                    else self.missing_embeddings[i]
                )
                for i in range(self.num_indices)
            ]
            result[batch_idx] = cat(current_embeddings, 0)

        return result

    @property
    def output_dim(self) -> int:
        return self.input_dim * self.num_indices


class SpanMeanPooler(nn.Module):
    """Pooler that takes the mean hidden state over spans. If the start or end index is negative, a
    learned embedding is used. The indices are expected to have the shape [batch_size,
    num_indices].

    The resulting embeddings are concatenated, so the output shape is [batch_size, num_indices * input_dim].
    Note this a slightly modified version of the pytorch_ie.models.components.pooler.SpanMaxPooler,
    i.e. we changed the aggregation method from torch.amax to torch.mean.

    Args:
        input_dim: The input dimension of the hidden state.
        num_indices: The number of indices to pool.

    Returns:
        The pooled hidden states with shape [batch_size, num_indices * input_dim].
    """

    def __init__(self, input_dim: int, num_indices: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.num_indices = num_indices
        self.missing_embeddings = nn.Parameter(torch.empty(num_indices, self.input_dim))
        nn.init.normal_(self.missing_embeddings)

    def forward(
        self, hidden_state: Tensor, start_indices: Tensor, end_indices: Tensor, **kwargs
    ) -> Tensor:
        batch_size, seq_len, hidden_size = hidden_state.shape
        if start_indices.shape[1] != self.num_indices:
            raise ValueError(
                f"number of start indices [{start_indices.shape[1]}] has to be the same as num_types [{self.num_indices}]"
            )

        if end_indices.shape[1] != self.num_indices:
            raise ValueError(
                f"number of end indices [{end_indices.shape[1]}] has to be the same as num_types [{self.num_indices}]"
            )

        # check that start_indices are before end_indices
        mask_both_positive = (start_indices >= 0) & (end_indices >= 0)
        mask_start_before_end = start_indices < end_indices
        mask_valid = mask_start_before_end | ~mask_both_positive
        if not torch.all(mask_valid):
            raise ValueError(
                f"values in start_indices have to be smaller than respective values in "
                f"end_indices, but start_indices=\n{start_indices}\n and end_indices=\n{end_indices}"
            )

        # times num_indices due to concat
        result = torch.zeros(
            batch_size, hidden_size * self.num_indices, device=hidden_state.device
        )
        for batch_idx in range(batch_size):
            current_start_indices = start_indices[batch_idx]
            current_end_indices = end_indices[batch_idx]
            current_embeddings = [
                (
                    torch.mean(
                        hidden_state[
                            batch_idx, current_start_indices[i] : current_end_indices[i], :
                        ],
                        dim=0,
                    )
                    if current_start_indices[i] >= 0 and current_end_indices[i] >= 0
                    else self.missing_embeddings[i]
                )
                for i in range(self.num_indices)
            ]
            result[batch_idx] = cat(current_embeddings, 0)

        return result

    @property
    def output_dim(self) -> int:
        return self.input_dim * self.num_indices


def get_pooler_and_output_size(config: Dict[str, Any], input_dim: int) -> Tuple[Callable, int]:
    pooler_config = dict(config)
    pooler_type = pooler_config.pop("type", CLS_TOKEN)
    pooler: Callable
    if pooler_type == CLS_TOKEN:
        return pool_cls, input_dim
    elif pooler_type == START_TOKENS:
        pooler = AtIndexPooler(input_dim=input_dim, offset=-1, **pooler_config)
        pooler_wrapped = ArgumentWrappedPooler(
            pooler=pooler, argument_mapping={"start_indices": "indices"}
        )
        return pooler_wrapped, pooler.output_dim
    elif pooler_type == MENTION_POOLING:
        aggregate = pooler_config.pop("aggregate", "max")
        if aggregate == "max":
            pooler = SpanMaxPooler(input_dim=input_dim, **pooler_config)
            return pooler, pooler.output_dim
        elif aggregate == "mean":
            pooler = SpanMeanPooler(input_dim=input_dim, **pooler_config)
            return pooler, pooler.output_dim
        else:
            raise ValueError(f'Unknown aggregation method for mention pooling: "{aggregate}"')
    else:
        raise ValueError(f'Unknown pooler type "{pooler_type}"')
