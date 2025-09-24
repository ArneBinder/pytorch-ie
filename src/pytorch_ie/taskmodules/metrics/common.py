import logging
from abc import ABC
from typing import Dict, Optional

import torch
from torch import LongTensor, Tensor
from torchmetrics import Metric

logger = logging.getLogger(__name__)


class MetricWithArbitraryCounts(Metric, ABC):
    """A metric that hold counts for arbitrary keys."""

    def inc_counts(self, counts: LongTensor, key: Optional[str], prefix: str = "counts_"):
        full_key = prefix
        if key is not None:
            full_key += key

        if not hasattr(self, full_key):
            self.add_state(full_key, default=torch.zeros_like(counts), dist_reduce_fx="sum")

        prev_value = getattr(self, full_key)
        setattr(self, full_key, prev_value + counts)

    def get_counts(self, key_prefix: str = "counts_") -> Dict[Optional[str], LongTensor]:
        result: Dict[Optional[str], LongTensor] = {}
        for k, v in self.metric_state.items():
            if k.startswith(key_prefix):
                if not isinstance(v, Tensor):
                    raise ValueError(
                        f"Expected metric state for key {k} to be a LongTensor, but got {type(v)}."
                    )
                if not isinstance(v, LongTensor):
                    v = v.long()
                assert isinstance(v, LongTensor)
                key = k[len(key_prefix) :] or None
                result[key] = v
        return result
