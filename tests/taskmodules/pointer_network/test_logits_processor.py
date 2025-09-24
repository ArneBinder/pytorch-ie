import pytest
import torch

from pytorch_ie.taskmodules.pointer_network.logits_processor import (
    FinitizeLogitsProcessor,
    PrefixConstrainedLogitsProcessorWithMaximum,
)


def test_prefix_constrained_logits_processor_with_maximum():
    def allow_last_three(batch_id, sent, max_index):
        return list(range(max_index - 3, max_index))

    logits_processor = PrefixConstrainedLogitsProcessorWithMaximum(
        prefix_allowed_tokens_fn=allow_last_three, num_beams=1
    )

    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]]).to(dtype=torch.long)
    scores = torch.tensor([[0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.0]]).to(dtype=torch.float)
    new_scores = logits_processor(input_ids, scores)
    assert new_scores.shape == scores.shape
    torch.testing.assert_close(
        new_scores,
        torch.tensor(
            [[-float("inf"), -float("inf"), -float("inf"), -float("inf"), 0.9, 0.9, 0.0]]
        ),
    )


def test_prefix_constrained_logits_processor_with_maximum_with_inf_scores():
    def allow_last_three(batch_id, sent, max_index):
        return list(range(max_index - 3, max_index))

    logits_processor = PrefixConstrainedLogitsProcessorWithMaximum(
        prefix_allowed_tokens_fn=allow_last_three, num_beams=1
    )
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]]).to(dtype=torch.long)
    scores_with_pos_inf = torch.tensor([[0.9, 0.9, float("inf"), 0.9, 0.9, 0.9, 0.0]]).to(
        dtype=torch.float
    )
    scores_with_neg_inf = torch.tensor([[0.9, 0.9, -float("inf"), 0.9, 0.9, 0.9, 0.0]]).to(
        dtype=torch.float
    )

    with pytest.raises(ValueError, match="scores contains ±inf or NaN"):
        logits_processor(input_ids, scores_with_pos_inf)

    with pytest.raises(ValueError, match="scores contains ±inf or NaN"):
        logits_processor(input_ids, scores_with_neg_inf)


def test_prefix_constrained_logits_processor_with_maximum_without_allowed_tokens():
    def allow_no_tokens(batch_id, sent, max_index):
        return []

    logits_processor = PrefixConstrainedLogitsProcessorWithMaximum(
        prefix_allowed_tokens_fn=allow_no_tokens, num_beams=1
    )

    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]]).to(dtype=torch.long)
    scores = torch.tensor([[0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.0]]).to(dtype=torch.float)

    with pytest.raises(ValueError, match="No allowed token ids for batch_id"):
        logits_processor(input_ids, scores)


def test_finitize_logits_processor():
    logits_processor = FinitizeLogitsProcessor()

    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]]).to(dtype=torch.long)
    scores = torch.tensor([[0.9, 0.9, float("inf"), 0.9, 0.9, -float("inf"), 0.0]]).to(
        dtype=torch.float
    )
    new_scores = logits_processor(input_ids, scores)

    assert new_scores.shape == scores.shape
    torch.testing.assert_close(
        new_scores,
        torch.tensor([[0.9, 0.9, 3.4028235e38, 0.9, 0.9, -3.4028235e38, 0.0]]),
    )
