from typing import Dict, List, Set, Tuple

import torch
import torch.nn.functional as F
from torchmetrics.metric import Metric


class SetFbetaScore(Metric):
    def __init__(self, none_index: int, beta: float = 1.0, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.none_index = none_index
        self.beta = beta

        self.add_state("true_positives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_positives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_negatives", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, List[torch.Tensor]]):
        label_ids_target = [target.detach().cpu() for target in targets["label_ids"]]
        start_index_target = [target.detach().cpu() for target in targets["start_index"]]
        end_index_target = [target.detach().cpu() for target in targets["end_index"]]

        label_ids_pred_full = F.softmax(predictions["label_ids"], dim=-1).argmax(dim=-1)
        start_index_pred_full = F.softmax(predictions["start_index"], dim=-1).argmax(dim=-1)
        end_index_pred_full = F.softmax(predictions["end_index"], dim=-1).argmax(dim=-1)

        for batch_index, (label_ids, start_index, end_index) in enumerate(
            zip(label_ids_target, start_index_target, end_index_target)
        ):
            pred_label_set: Set[Tuple[int, int, int]] = set()
            true_label_set: Set[Tuple[int, int, int]] = set()

            indices_pred = label_ids_pred_full[batch_index] != self.none_index
            label_ids_pred = label_ids_pred_full[batch_index][indices_pred].detach().cpu()
            start_index_pred = start_index_pred_full[batch_index][indices_pred].detach().cpu()
            end_index_pred = end_index_pred_full[batch_index][indices_pred].detach().cpu()

            for i in range(label_ids_pred.shape[0]):
                pred_label_set.add(
                    (
                        label_ids_pred[i].item(),
                        start_index_pred[i].item(),
                        end_index_pred[i].item(),
                    )
                )

            for i in range(label_ids.shape[0]):
                true_label_set.add(
                    (label_ids[i].item(), start_index[i].item(), end_index[i].item())
                )

            for pred in pred_label_set:
                if pred in true_label_set:
                    self.true_positives += 1
                else:
                    self.false_positives += 1

            for pred in true_label_set:
                if pred not in pred_label_set:
                    self.false_negatives += 1

    def compute(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-10)
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-10)
        f1 = self.beta * precision * recall / (precision + recall + 1e-10)

        return f1
