from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from pytorch_ie.models.set_prediction.matching.cost_functions import CostFunction


class HungarianMatcher(nn.Module):
    def __init__(
        self,
        cost_functions: Dict[str, CostFunction],
        cost_weights: Dict[str, float],
        target_labels_attribute: str = "label_ids",
    ) -> None:
        super().__init__()
        self.cost_functions = nn.ModuleDict(cost_functions)
        self.cost_weights = cost_weights
        # TODO: make this more generic, e.g., select first key?
        self.target_labels_attribute = target_labels_attribute

    @torch.no_grad()
    def forward(
        self,
        output: Dict[str, torch.Tensor],
        targets: Dict[str, List[torch.Tensor]],
        prev_permutation_indices: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        costs: Optional[torch.Tensor] = None
        for name, cost_function in self.cost_functions.items():
            cost = cost_function(name, output, targets, prev_permutation_indices)
            if cost is None:
                continue

            cost = cost * self.cost_weights.get(name, 1.0)

            if costs is None:
                costs = cost
            else:
                costs += cost

        if costs is None:
            raise ValueError("not any cost was calculated")

        costs = costs.cpu()  # [batch_size, num_queries, num_targets_total]

        sizes = [len(target) for target in targets[self.target_labels_attribute]]

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(costs.split(sizes, -1))]
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]
