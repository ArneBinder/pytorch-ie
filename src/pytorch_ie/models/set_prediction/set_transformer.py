from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from transformers import PreTrainedModel

from pytorch_ie.models.set_prediction.loss.set_criterion import SetCriterion
from pytorch_ie.models.set_prediction.matching.matcher import HungarianMatcher
from pytorch_ie.models.set_prediction.set_decoder import SetDecoder


class SetTransformer(nn.Module):
    def __init__(
        self,
        encoder: PreTrainedModel,
        set_decoders: List[Tuple[str, SetDecoder]],
        matchers: Dict[str, HungarianMatcher],
        set_criteria: Dict[str, SetCriterion],
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.set_decoder_names = [name for name, _ in set_decoders]
        self.set_decoders = nn.ModuleDict(set_decoders)

        # TODO: add check for matcher and set_criteria
        # if not all([name in matchers for name in self.set_decoder_names]):
        #     pass

        self.matchers = nn.ModuleDict(matchers)
        self.set_criteria = nn.ModuleDict(set_criteria)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=False,
        output_hidden_states=False,
    ) -> Dict[str, torch.Tensor]:
        outputs = {}

        encoder_outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=False,
            output_hidden_states=False,
        )
        hidden_states_encoder = encoder_outputs[0]  # [batch_size, input_length, hidden_size]

        prev_output = hidden_states_encoder
        prev_output_attention_mask = attention_mask
        for name in self.set_decoder_names:
            decoder = self.set_decoders[name]
            queries = decoder.get_queries(num_batch=input_ids.shape[0])
            queries_attention_mask = None
            output, set_states = decoder(
                queries=queries,
                queries_attention_mask=queries_attention_mask,
                prev_output=prev_output,
                prev_output_attention_mask=prev_output_attention_mask,
            )
            outputs[name] = output

            prev_output = set_states
            prev_output_attention_mask = queries_attention_mask

        return outputs

    def get_loss(
        self, outputs: Dict[str, Dict[str, torch.Tensor]], targets: Dict[str, Dict[str, List[torch.Tensor]]]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        losses: Dict[str, Dict[str, float]] = {}
        prev_permutation_indices = None
        for name in self.set_decoder_names:
            matcher = self.matchers[name]
            criterion = self.set_criteria[name]
            permutation_indices = matcher(
                output=outputs[name],
                targets=targets[name],
                prev_permutation_indices=prev_permutation_indices,
            )
            loss = criterion(
                output=outputs[name],
                targets=targets[name],
                permutation_indices=permutation_indices,
                prev_permutation_indices=prev_permutation_indices,
            )
            losses[name] = loss

            prev_permutation_indices = permutation_indices

        return losses
