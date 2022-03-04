from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SetDecoder(nn.Module):
    def __init__(
        self,
        config: BertConfig,
        num_queries: int,
        num_labels: int,
        query_prototypes: str = "learned",
    ) -> None:
        super().__init__()
        if not config.is_decoder:
            raise ValueError("is_decoder must be set to true in config.")

        self.decoder = BertEncoder(config)
        self.decoder.apply(self._init_decoder_weights)

        self.num_queries = num_queries
        self.num_labels = num_labels
        self.query_prototypes = query_prototypes.lower()

        hidden_size = self.decoder.config.hidden_size
        # self.class_embed = MLP(
        #     input_dim=hidden_size,
        #     hidden_dim=hidden_size // 2,
        #     output_dim=self.num_labels + 1,  # Add one more label for "not a set item"
        #     num_layers=2,
        # )
        # self.class_embed = nn.Linear(
        #     hidden_size, self.num_labels + 1
        # )  # Add one more label for "not a set item"

        self.class_embed = MLP(
            input_dim=hidden_size,
            hidden_dim=hidden_size,
            output_dim=self.num_labels + 1,
            num_layers=2,
        )  # Add one more label for "not a set item"

        if query_prototypes == "learned":
            self.query_embed = nn.Embedding(self.num_queries, self.decoder.config.hidden_size)
        else:
            raise NotImplementedError()

    def _init_decoder_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.decoder.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.decoder.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_queries(self, num_batch: int, num_queries: Optional[int] = None) -> torch.Tensor:
        if self.query_prototypes == "learned":
            # TODO: use expand instead of repeat and test
            return self.query_embed.weight.unsqueeze(0).repeat(num_batch, 1, 1)
        else:
            raise NotImplementedError()

    def decode(
        self,
        set_states: torch.Tensor,
        prev_output: torch.Tensor,
        prev_output_attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return {}

    def forward(
        self,
        queries,
        prev_output,
        queries_attention_mask=None,
        prev_output_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
    ) -> Tuple[Dict[str, Any], torch.Tensor]:
        decoder_head_mask = [None] * self.decoder.config.num_hidden_layers

        # if queries_attention_mask is None:
        #     queries_attention_mask = torch.ones(queries.shape[:-1], device=queries.device)

        # if prev_output_attention_mask is None:
        #     prev_output_attention_mask = torch.ones(prev_output.shape[:-1], device=prev_output.device)

        # If a 2d or 3d attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if prev_output_attention_mask is not None:
            if prev_output_attention_mask.dim() == 3:
                prev_output_attention_mask = prev_output_attention_mask[:, None, :, :]
            if prev_output_attention_mask.dim() == 2:
                prev_output_attention_mask = prev_output_attention_mask[:, None, None, :]
            # prev_extended_attention_mask = prev_extended_attention_mask.to(
            #     dtype=self.dtype
            # )  # fp16 compatibility

        # print("queries: ", queries.shape)
        # print("queries_attention_mask: ", queries_attention_mask)
        # print("prev_output_attention_mask: ", prev_output_attention_mask)
        # print("prev_output: ", prev_output.shape)

        decoder_outputs = self.decoder(
            queries,
            attention_mask=queries_attention_mask,
            head_mask=decoder_head_mask,
            encoder_hidden_states=prev_output,
            encoder_attention_mask=prev_output_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        set_states = decoder_outputs[0]  # [batch_size, num_queries, hidden_size]

        output = {
            "label_ids": self.class_embed(set_states)  # [batch_size, num_queries, num_labels + 1]
        }

        output.update(self.decode(set_states, prev_output, prev_output_attention_mask))

        return output, set_states


class SpanLabelDecoder(SetDecoder):
    def __init__(
        self,
        config: BertConfig,
        num_queries: int,
        num_labels: int,
        query_prototypes: str = "learned",
    ) -> None:
        super().__init__(config, num_queries, num_labels, query_prototypes)

        hidden_size = self.decoder.config.hidden_size
        self.start_embed = MLP(
            input_dim=hidden_size,
            hidden_dim=hidden_size,
            output_dim=hidden_size,
            num_layers=2,
        )
        self.end_embed = MLP(
            input_dim=hidden_size,
            hidden_dim=hidden_size,
            output_dim=hidden_size,
            num_layers=2,
        )
        self.span_position_embed = MLP(
            input_dim=hidden_size,
            hidden_dim=hidden_size,
            output_dim=3,
            num_layers=2,
        )  # predict [span_start, span_end, span_width] normalized in [0, 1]

    def decode(
        self, set_states, prev_output, prev_output_attention_mask
    ) -> Dict[str, torch.Tensor]:
        # apply mask and set excluded positions to a large negative number (see transformers)
        mask = prev_output_attention_mask.squeeze(1)
        mask = (1.0 - mask) * -10000.0

        pred_start_index = torch.einsum(
            "bij,bkj->bik", self.start_embed(set_states), self.start_embed(prev_output)
        )  # [batch_size, num_queries, input_length]
        pred_start_index = pred_start_index + mask

        pred_end_index = torch.einsum(
            "bij,bkj->bik", self.end_embed(set_states), self.end_embed(prev_output)
        )  # [batch_size, num_queries, input_length]
        pred_end_index = pred_end_index + mask

        pred_span_position = self.span_position_embed(set_states).sigmoid()

        return {
            "start_index": pred_start_index,
            "end_index": pred_end_index,
            "span_position": pred_span_position,
        }


class SpanLabelJointDecoder(SetDecoder):
    def __init__(
        self,
        config: BertConfig,
        num_queries: int,
        num_labels: int,
        query_prototypes: str = "learned",
    ) -> None:
        super().__init__(config, num_queries, num_labels, query_prototypes)

        hidden_size = self.decoder.config.hidden_size
        self.start_embed = MLP(
            input_dim=2 * hidden_size,
            hidden_dim=hidden_size,
            output_dim=1,
            num_layers=2,
        )
        self.end_embed = MLP(
            input_dim=2 * hidden_size,
            hidden_dim=hidden_size,
            output_dim=1,
            num_layers=2,
        )
        self.span_position_embed = MLP(
            input_dim=hidden_size,
            hidden_dim=hidden_size,
            output_dim=3,
            num_layers=2,
        )  # predict [span_start, span_end, span_width] normalized in [0, 1]
        self.span_embed = MLP(
            input_dim=2 * hidden_size,
            hidden_dim=hidden_size,
            output_dim=1,
            num_layers=2,
        )

    def decode(
        self, set_states, prev_output, prev_output_attention_mask
    ) -> Dict[str, torch.Tensor]:
        # apply mask and set excluded positions to a large negative number (see transformers)
        mask = prev_output_attention_mask.squeeze(1)
        mask = (1.0 - mask) * -10000.0

        seq_len = prev_output.shape[1]
        batch_size, num_queries, hidden_size = set_states.shape
        head = set_states.unsqueeze(2).expand(batch_size, num_queries, seq_len, hidden_size)
        tail = prev_output.unsqueeze(1).expand(batch_size, num_queries, seq_len, hidden_size)
        pred_start_index = self.start_embed(torch.cat([head, tail], dim=-1)).squeeze(-1)
        # pred_start_index = torch.einsum(
        #     "bij,bkj->bik", self.start_embed(set_states), self.start_embed(prev_output)
        # )  # [batch_size, num_queries, input_length]
        pred_start_index = pred_start_index + mask

        pred_end_index = self.end_embed(torch.cat([head, tail], dim=-1)).squeeze(-1)
        # pred_end_index = torch.einsum(
        #     "bij,bkj->bik", self.end_embed(set_states), self.end_embed(prev_output)
        # )  # [batch_size, num_queries, input_length]
        pred_end_index = pred_end_index + mask

        pred_span_position = self.span_position_embed(set_states).sigmoid()

        pred_span_mask = self.span_embed(torch.cat([head, tail], dim=-1)).squeeze(-1)
        # [batch_size, num_queries, input_length]
        pred_span_mask = pred_span_mask + mask

        return {
            "start_index": pred_start_index,
            "end_index": pred_end_index,
            "span_position": pred_span_position,
            "span_mask": pred_span_mask,
        }


class SpanLabelAndMaskDecoder(SetDecoder):
    def __init__(
        self,
        config: BertConfig,
        num_queries: int,
        num_labels: int,
        query_prototypes: str = "learned",
    ) -> None:
        super().__init__(config, num_queries, num_labels, query_prototypes)

        hidden_size = self.decoder.config.hidden_size
        self.start_embed = MLP(
            input_dim=hidden_size,
            hidden_dim=hidden_size,
            output_dim=hidden_size,
            num_layers=2,
        )
        self.end_embed = MLP(
            input_dim=hidden_size,
            hidden_dim=hidden_size,
            output_dim=hidden_size,
            num_layers=2,
        )
        self.span_position_embed = MLP(
            input_dim=hidden_size,
            hidden_dim=hidden_size,
            output_dim=3,
            num_layers=2,
        )  # predict [span_start, span_end, span_width] normalized in [0, 1]
        self.mask_embed = MLP(
            input_dim=hidden_size,
            hidden_dim=hidden_size,
            output_dim=1,
            num_layers=2,
        )

    def decode(
        self, set_states, prev_output, prev_output_attention_mask
    ) -> Dict[str, torch.Tensor]:
        # apply mask and set excluded positions to a large negative number (see transformers)
        mask = prev_output_attention_mask.squeeze(1)
        mask = (1.0 - mask) * -10000.0

        pred_start_index = torch.einsum(
            "bij,bkj->bik", self.start_embed(set_states), self.start_embed(prev_output)
        )  # [batch_size, num_queries, input_length]
        pred_start_index = pred_start_index + mask

        pred_end_index = torch.einsum(
            "bij,bkj->bik", self.end_embed(set_states), self.end_embed(prev_output)
        )  # [batch_size, num_queries, input_length]
        pred_end_index = pred_end_index + mask

        pred_span_position = self.span_position_embed(set_states).sigmoid()

        pred_span_mask = torch.einsum(
            "bij,bkj->bik", self.mask_embed(set_states), self.mask_embed(prev_output)
        )  # [batch_size, num_queries, input_length]

        return {
            "start_index": pred_start_index,
            "end_index": pred_end_index,
            "span_position": pred_span_position,
            "span_mask": pred_span_mask,
        }
