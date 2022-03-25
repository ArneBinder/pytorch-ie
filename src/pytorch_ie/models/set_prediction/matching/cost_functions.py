from itertools import chain
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class CostFunction(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        name: str,
        output: Dict[str, torch.Tensor],
        targets: Dict[str, List[torch.Tensor]],
        prev_permutation_indices: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Optional[torch.Tensor]:
        raise NotImplementedError()


class CrossEntropyCostFunction(CostFunction):
    def forward(
        self,
        name: str,
        output: Dict[str, torch.Tensor],
        targets: Dict[str, List[torch.Tensor]],
        prev_permutation_indices: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Optional[torch.Tensor]:
        logits = output[name]
        batch_size, num_queries = logits.shape[:2]

        probabilities = logits.flatten(0, 1).softmax(
            dim=-1
        )  # [batch_size * num_queries, num_classes]

        target_indices = torch.cat(targets[name]).long()  # [num_targets_total]

        cost_class = -probabilities[
            :, target_indices
        ]  # [batch_size * num_queries, num_targets_total]

        cost_class = cost_class.view(
            batch_size, num_queries, -1
        )  # [batch_size, num_queries, num_targets_total]

        return cost_class


class BinaryCrossEntropyCostFunction(CostFunction):
    def forward(
        self,
        name: str,
        output: Dict[str, torch.Tensor],
        targets: Dict[str, List[torch.Tensor]],
        prev_permutation_indices: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Optional[torch.Tensor]:
        logits = output[name]
        batch_size, num_queries = logits.shape[:2]

        out_token_mask = logits.flatten(0, 1)  # [batch_size * num_queries, seq_len]

        tgt_token_mask = torch.cat(targets[name], dim=0)  # [num_targets_total, seq_len]

        out_token_mask = out_token_mask.unsqueeze(1).expand(-1, tgt_token_mask.size(0), -1)
        tgt_token_mask = (
            tgt_token_mask.unsqueeze(0).expand_as(out_token_mask).to(out_token_mask.device)
        )

        mask = tgt_token_mask != -100
        cost_span = F.binary_cross_entropy_with_logits(
            out_token_mask, tgt_token_mask.float(), reduction="none"
        )
        cost_span = ((cost_span * mask) / mask.sum(dim=-1, keepdims=True)).sum(dim=-1)

        cost_span = cost_span.view(
            batch_size, num_queries, -1
        )  # [batch_size, num_queries, num_targets_total]

        return cost_span


class SpanPositionCostFunction(CostFunction):
    def forward(
        self,
        name: str,
        output: Dict[str, torch.Tensor],
        targets: Dict[str, List[torch.Tensor]],
        prev_permutation_indices: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Optional[torch.Tensor]:
        if len(targets[name]) <= 0:
            return None

        positions = output[name]  # [batch_size, num_queries, 3]
        batch_size, num_queries = positions.shape[:2]

        positions = positions.flatten(0, 1)  # [batch_size * num_queries, 3]
        target_positions = torch.cat(targets[name])

        cost_position = torch.cdist(positions, target_positions, p=1)

        cost_position = cost_position.view(
            batch_size, num_queries, -1
        )  # [batch_size, num_queries, num_targets_total]

        return cost_position


class EdgeCostFunction(CostFunction):
    def forward(
        self,
        name: str,
        output: Dict[str, torch.Tensor],
        targets: Dict[str, List[torch.Tensor]],
        prev_permutation_indices: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Optional[torch.Tensor]:
        batch_edges = output[name]

        batch_edge_probabilities = batch_edges.softmax(
            dim=-1
        )  # [batch_size, num_queries, num_prev_queries, num_classes]

        batch_size, num_queries = batch_edges.shape[:2]

        sorted_query_indices = [
            query_indices[torch.argsort(target_indices)]
            for query_indices, target_indices in prev_permutation_indices
        ]

        batch_cost_class: List[torch.Tensor] = []
        for batch_index, (edges, prev_query_indices, edge_target_indices, col_ids) in enumerate(
            zip(
                batch_edge_probabilities,
                sorted_query_indices,
                targets["edge_ids"],
                targets["col_ids"],
            )
        ):
            if len(edge_target_indices) <= 0:
                continue

            valid_edges = edges[
                :, prev_query_indices, :
            ]  # [num_queries, num_prev_targets, num_classes]

            col_indices = torch.cat(
                [col_i.long() for col_i in col_ids], dim=0
            )  # [num_edge_targets]

            all_edges = valid_edges[
                :, col_indices, :
            ]  # [num_queries, num_edge_targets, num_classes]

            sizes = [len(col_i) for col_i in col_ids]

            cost_splits: List[torch.Tensor] = []
            for edge_split, target_split in zip(
                torch.split(all_edges, sizes, dim=1), edge_target_indices
            ):
                ind = torch.arange(end=edge_split.shape[1], device=edge_split.device)
                neg_probs = -edge_split[:, ind, target_split.long()]
                cost_splits.append(torch.sum(neg_probs, dim=-1, keepdim=True))  # [num_queries, 1]

            cost_class = torch.cat(cost_splits, dim=-1)  # [num_queries, num_targets]

            batch_cost_class.append(cost_class)

        if len(batch_cost_class) <= 0:
            return None

        batch_cost_class = torch.cat(batch_cost_class, dim=-1).expand(
            batch_size, -1, -1
        )  # [batch_size, num_queries, num_targets_total]

        return batch_cost_class
