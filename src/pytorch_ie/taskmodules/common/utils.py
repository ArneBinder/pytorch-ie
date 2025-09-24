import logging
from typing import Union

import torch

logger = logging.getLogger(__name__)


def get_first_occurrence_index(
    tensor: Union[torch.FloatTensor, torch.LongTensor], value: Union[float, int]
) -> torch.LongTensor:
    """Returns the index of the first occurrence of `value` in each row of `tensor`. If `value` is
    not found, seq_len is returned.

    Args:
        tensor: the tensor of shape (bsz, seq_len) to search in
        value: the value to search for

    Returns: a tensor of shape (bsz,) containing the index of the first occurrence of `value` in each row of `tensor`.
    """

    mask_value = tensor.eq(value)
    # count matching positions from the end
    value_counts_to_end = mask_value.flip(dims=[1]).cumsum(dim=1).flip(dims=[1])
    # at the first position stands the number of total matches
    total_matches = value_counts_to_end[:, 0]
    # the sum of all positions where the number of matches is equal to the total number of matches
    # is the index *after* the first occurrence
    result = value_counts_to_end.eq(total_matches.unsqueeze(-1)).sum(dim=1) - 1
    # set result to seq_len if no match was found
    result[total_matches == 0] = tensor.size(1)

    assert isinstance(result, torch.LongTensor)
    return result
