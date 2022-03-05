from typing import Dict, List, Tuple

from torch import nn, Tensor

from pytorch_ie.models.set_prediction.loss.loss_functions import LossFunction


class SetCriterion(nn.Module):
    def __init__(
        self,
        loss_functions: Dict[str, LossFunction],
        loss_weights: Dict[str, float],
    ) -> None:
        super().__init__()
        self.loss_functions = nn.ModuleDict(loss_functions)
        self.loss_weights = loss_weights

    def forward(
        self,
        output: Dict[str, Tensor],
        targets: Dict[str, List[Tensor]],
        permutation_indices: Tuple[Tensor, Tensor],
        prev_permutation_indices: Tuple[Tensor, Tensor],
    ) -> Dict[str, Tensor]:
        loss: Dict[str, Tensor] = {}
        for name, loss_function in self.loss_functions.items():
            loss[name] = loss_function(
                name, output, targets, permutation_indices, prev_permutation_indices
            ) * self.loss_weights.get(name, 1.0)

        return loss
