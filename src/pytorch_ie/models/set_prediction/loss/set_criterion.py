from typing import Any, Callable, Dict, List, Tuple

import torch
from torch import nn

from pytorch_ie.models.set_prediction.loss.loss_functions import LossFunction


class SetCriterion(nn.Module):
    def __init__(
        self,
        loss_functions: Dict[str, LossFunction],
        loss_weights: Dict[str, int],
    ) -> None:
        super().__init__()
        self.loss_functions = nn.ModuleDict(loss_functions)
        self.loss_weights = loss_weights

    def forward(
        self,
        output: Dict[str, torch.Tensor],
        targets: List[Dict[str, Any]],
        permutation_indices: Tuple[torch.Tensor, torch.Tensor],
        prev_permutation_indices: Tuple[torch.Tensor, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        loss: Dict[str, torch.Tensor] = {}
        for name, loss_function in self.loss_functions.items():
            loss[name] = loss_function(
                name, output, targets, permutation_indices, prev_permutation_indices
            ) * self.loss_weights.get(name, 1.0)

        return loss
