from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


def get_tgt_permutation_idx(indices):
    # permute targets following indices
    batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
    tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    return batch_idx, tgt_idx


class LossFunction(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        name: str,
        output: Dict[str, torch.Tensor],
        targets: Dict[str, List[torch.Tensor]],
        permutation_indices: Tuple[torch.Tensor, torch.Tensor],
        prev_permutation_indices: Tuple[torch.Tensor, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        raise NotImplementedError()


class CrossEntropyLossFunction(LossFunction):
    def __init__(
        self, num_classes: Optional[int] = None, none_weight: float = 1.0, none_index: int = 0
    ):
        super().__init__()
        if num_classes is not None:
            empty_weight = torch.ones(num_classes + 1)
            empty_weight[none_index] = none_weight
            self.register_buffer("empty_weight", empty_weight)
        else:
            self.empty_weight = None

        self.none_index = none_index

    def forward(
        self,
        name: str,
        output: Dict[str, torch.Tensor],
        targets: Dict[str, List[torch.Tensor]],
        permutation_indices: Tuple[torch.Tensor, torch.Tensor],
        prev_permutation_indices: Tuple[torch.Tensor, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        src_logits = output[name]  # [batch_size, num_queries, num_targets]

        idx = get_src_permutation_idx(permutation_indices)
        idx = idx[0].to(src_logits.device), idx[1].to(src_logits.device)

        target_classes_o = torch.cat(
            [target[i] for target, (_, i) in zip(targets[name], permutation_indices)], dim=0
        ).long()
        target_classes_o = target_classes_o.to(src_logits.device)

        target_classes = torch.full(
            src_logits.shape[:2], self.none_index, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)

        return loss


class BinaryCrossEntropyLossFunction(LossFunction):
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(
        self,
        name: str,
        output: Dict[str, torch.Tensor],
        targets: Dict[str, List[torch.Tensor]],
        permutation_indices: Tuple[torch.Tensor, torch.Tensor],
        prev_permutation_indices: Tuple[torch.Tensor, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        src_token_mask = output[name]  # [batch_size, num_queries, seq_len]

        idx = get_src_permutation_idx(permutation_indices)
        idx = idx[0].to(src_token_mask.device), idx[1].to(src_token_mask.device)

        src_token_mask = output[name][idx]

        target_token_mask = torch.cat(
            [target[i] for target, (_, i) in zip(targets[name], permutation_indices)], dim=0
        ).float()
        target_token_mask = target_token_mask.to(src_token_mask.device)

        mask = target_token_mask != -100
        loss_span = F.binary_cross_entropy_with_logits(
            src_token_mask, target_token_mask.float(), reduction="none"
        )

        loss = ((loss_span * mask) / mask.sum(dim=-1, keepdims=True)).sum()

        return loss


class SpanPositionLossFunction(LossFunction):
    def forward(
        self,
        name: str,
        output: Dict[str, torch.Tensor],
        targets: Dict[str, List[torch.Tensor]],
        permutation_indices: Tuple[torch.Tensor, torch.Tensor],
        prev_permutation_indices: Tuple[torch.Tensor, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if len(targets[name]) <= 0:
            return None

        idx = get_src_permutation_idx(permutation_indices)
        src_span_positions = output[name][idx]
        target_span_positions = torch.cat(
            [target[i] for target, (_, i) in zip(targets[name], permutation_indices)], dim=0
        )

        loss = F.l1_loss(src_span_positions, target_span_positions)

        return loss


class EdgeLossFunction(CrossEntropyLossFunction):
    def forward(
        self,
        name: str,
        output: Dict[str, torch.Tensor],
        targets: Dict[str, List[torch.Tensor]],
        permutation_indices: Tuple[torch.Tensor, torch.Tensor],
        prev_permutation_indices: Tuple[torch.Tensor, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        batch_edges = output[name]  # [batch_size, num_queries, num_prev_queries, num_classes]

        batch_size, num_queries = batch_edges.shape[:2]

        sorted_prev_query_indices = [
            query_indices[torch.argsort(target_indices)]
            for query_indices, target_indices in prev_permutation_indices
        ]

        sorted_query_indices = [
            query_indices[torch.argsort(target_indices)]
            for query_indices, target_indices in permutation_indices
        ]

        total_loss: torch.Tensor = None
        for (
            batch_index,
            (edges, prev_query_indices, query_indices, edge_ids, row_ids, col_ids),
        ) in enumerate(
            zip(
                batch_edges,
                sorted_prev_query_indices,
                sorted_query_indices,
                targets["edge_ids"],
                targets["row_ids"],
                targets["col_ids"],
            )
        ):
            if len(edge_ids) <= 0:
                continue

            valid_edges = edges[:, prev_query_indices, :][
                query_indices, :, :
            ]  # [num_targets, num_prev_targets, num_classes]

            row_indices = torch.cat(
                [row_i.long() for row_i in row_ids], dim=0
            )  # [num_edge_targets]

            col_indices = torch.cat(
                [col_i.long() for col_i in col_ids], dim=0
            )  # [num_edge_targets]

            target_classes = torch.full(
                valid_edges.shape[:2],
                self.none_index,
                dtype=torch.int64,
                device=valid_edges.device,
            )

            edge_target_indices = torch.cat(
                [edge_i.long() for edge_i in edge_ids], dim=0
            )  # [num_edge_targets]

            target_classes[row_indices, col_indices] = edge_target_indices

            loss = F.cross_entropy(valid_edges.transpose(1, 2), target_classes, self.empty_weight)

            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss

        return total_loss
