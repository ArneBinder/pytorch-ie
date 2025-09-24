from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.utils import logging

logger = logging.get_logger(__name__)


class PointerHead(torch.nn.Module):
    # Copy and generate,
    def __init__(
        self,
        # (decoder) input space
        target_token_ids: List[int],
        # output space (targets)
        bos_id: int,
        eos_id: int,
        pad_id: int,
        # embeddings
        embeddings: nn.Embedding,
        embedding_weight_mapping: Optional[Dict[Union[int, str], List[int]]] = None,
        # other parameters
        use_encoder_mlp: bool = False,
        use_constraints_encoder_mlp: bool = False,
        decoder_position_id_mode: Optional[nn.Module] = None,
        decoder_position_id_pattern: Optional[List[int]] = None,
        decoder_position_id_mapping: Optional[Dict[str, int]] = None,
    ):
        super().__init__()

        self.embeddings = embeddings

        self.pointer_offset = len(target_token_ids)

        # check that bos, eos, and pad are not out of bounds
        for target_id, target_id_name in zip(
            [bos_id, eos_id, pad_id], ["bos_id", "eos_id", "pad_id"]
        ):
            if target_id >= len(target_token_ids):
                raise ValueError(
                    f"{target_id_name} [{target_id}] must be smaller than the number of target token ids "
                    f"[{len(target_token_ids)}]!"
                )

        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        # all ids that are not bos, eos or pad are label ids
        self.label_ids = [
            target_id
            for target_id in range(len(target_token_ids))
            if target_id not in [self.bos_id, self.eos_id, self.pad_id]
        ]

        target2token_id = torch.LongTensor(target_token_ids)
        self.register_buffer("target2token_id", target2token_id)
        self.label_token_ids = self.target2token_id[self.label_ids]
        self.eos_token_id = target_token_ids[self.eos_id]
        self.pad_token_id = target_token_ids[self.pad_id]

        hidden_size = self.embeddings.embedding_dim
        if use_encoder_mlp:
            self.encoder_mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Dropout(0.3),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
            )
        if use_constraints_encoder_mlp:
            self.constraints_encoder_mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Dropout(0.3),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
            )

        self.embedding_weight_mapping = None
        if embedding_weight_mapping is not None:
            # Because of config serialization, the keys may be strings. Convert them back to ints.
            self.embedding_weight_mapping = {
                int(k): v for k, v in embedding_weight_mapping.items()
            }

        self.decoder_position_id_mode = decoder_position_id_mode
        self.decoder_position_id_mapping = decoder_position_id_mapping
        if self.decoder_position_id_mode is None:
            pass
        elif self.decoder_position_id_mode in ["pattern", "pattern_with_increment"]:
            if decoder_position_id_pattern is None:
                raise ValueError(
                    "decoder_position_id_pattern must be provided when using "
                    'decoder_position_id_mode="pattern" or "pattern_with_increment"!'
                )
            self.register_buffer(
                "decoder_position_id_pattern", torch.tensor(decoder_position_id_pattern)
            )
        elif self.decoder_position_id_mode == "mapping":
            if self.decoder_position_id_mapping is None:
                raise ValueError(
                    'decoder_position_id_mode="mapping" requires decoder_position_id_mapping to be provided!'
                )
        else:
            raise ValueError(
                f'decoder_position_id_mode="{self.decoder_position_id_mode}" is not supported, '
                'use one of "pattern", "pattern_with_increment", or "mapping"!'
            )

    @property
    def use_prepared_position_ids(self):
        return self.decoder_position_id_mode is not None

    def set_embeddings(self, embedding: nn.Embedding) -> None:
        self.embeddings = embedding

    def overwrite_embeddings_with_mapping(self) -> None:
        """Overwrite individual embeddings with embeddings for other tokens.

        This is useful, for instance, if the label vocabulary is a subset of the source vocabulary.
        In this case, this method can be used to initialize each label embedding with one or
        multiple (averaged) source embeddings.
        """
        if self.embedding_weight_mapping is not None:
            for special_token_index, source_indices in self.embedding_weight_mapping.items():
                self.embeddings.weight.data[special_token_index] = self.embeddings.weight.data[
                    source_indices
                ].mean(dim=0)

    def prepare_decoder_input_ids(
        self,
        input_ids: torch.LongTensor,
        encoder_input_ids: torch.LongTensor,
    ) -> torch.LongTensor:
        mapping_token_mask = input_ids.lt(self.pointer_offset)
        mapped_tokens = input_ids.masked_fill(input_ids.ge(self.pointer_offset), 0)
        tag_mapped_tokens = self.target2token_id[mapped_tokens]

        encoder_input_ids_index = input_ids - self.pointer_offset
        encoder_input_ids_index = encoder_input_ids_index.masked_fill(
            encoder_input_ids_index.lt(0), 0
        )
        encoder_input_length = encoder_input_ids.size(1)
        if encoder_input_ids_index.max() >= encoder_input_length:
            raise ValueError(
                f"encoder_input_ids_index.max() [{encoder_input_ids_index.max()}] must be smaller "
                f"than encoder_input_length [{encoder_input_length}]!"
            )

        word_mapped_tokens = encoder_input_ids.gather(index=encoder_input_ids_index, dim=1)

        decoder_input_ids = torch.where(
            mapping_token_mask, tag_mapped_tokens, word_mapped_tokens
        ).to(torch.long)
        assert isinstance(decoder_input_ids, torch.LongTensor)

        # Note: we do not need to explicitly handle the padding (via a decoder attention mask) because
        # it gets automatically mapped to the pad token id

        return decoder_input_ids

    def prepare_decoder_position_ids(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        if self.decoder_position_id_mode in ["pattern", "pattern_with_increment"]:
            bsz, tokens_len = input_ids.size()
            pattern_len = len(self.decoder_position_id_pattern)
            # the number of full and partly records. note that tokens_len includes the bos token
            repeat_num = (tokens_len - 2) // pattern_len + 1
            position_ids = self.decoder_position_id_pattern.repeat(bsz, repeat_num)

            if self.decoder_position_id_mode == "pattern_with_increment":
                position_ids_reshaped = position_ids.view(bsz, -1, pattern_len)
                add_shift_pos = (
                    torch.arange(0, repeat_num, device=position_ids_reshaped.device)
                    .repeat(bsz)
                    .view(bsz, -1)
                    .unsqueeze(-1)
                )
                # multiply by the highest position id in the pattern so that the position ids are unique
                # for any decoder_position_id_pattern across all records
                add_shift_pos *= max(self.decoder_position_id_pattern) + 1
                position_ids_reshaped = add_shift_pos + position_ids_reshaped
                position_ids = position_ids_reshaped.view(bsz, -1).long()
            # use start_position_id=0
            start_pos = torch.zeros(bsz, 1, dtype=position_ids.dtype, device=position_ids.device)
            # shift by 2 to account for start_position_id=0 and pad_position_id=1
            all_position_ids = torch.cat([start_pos, position_ids + 2], dim=-1)
            all_position_ids_truncated = all_position_ids[:bsz, :tokens_len]

            # mask the padding tokens
            mask_invalid = input_ids.eq(self.pad_id)
            all_position_ids_truncated_masked = all_position_ids_truncated.masked_fill(
                mask_invalid, 1
            ).to(torch.long)
            assert isinstance(all_position_ids_truncated_masked, torch.LongTensor)
            return all_position_ids_truncated_masked
        elif self.decoder_position_id_mode == "mapping":
            # we ignor the typing issue here because we ensure that the mapping is not None in the __init__
            mapping: Dict[str, int] = self.decoder_position_id_mapping  # type: ignore
            if "default" not in mapping:
                raise ValueError(
                    f"mapping must contain a default entry, but only contains {list(mapping)}!"
                )
            position_ids = input_ids.new_full(input_ids.size(), fill_value=mapping["default"])
            # ensure that values for all vocab entries are set first
            if "vocab" in mapping:
                position_ids[input_ids.lt(self.pointer_offset)] = mapping["vocab"]
            already_set: Dict[int, Tuple[str, int]] = {}
            for key, value in mapping.items():
                if key in ["default", "vocab"]:
                    continue
                elif key == "bos":
                    input_id = self.bos_id
                elif key == "eos":
                    input_id = self.eos_id
                elif key == "pad":
                    input_id = self.pad_id
                else:
                    raise ValueError(f"Mapping contains unknown key '{key}' (mapping: {mapping}).")
                if already_set.get(input_id, (key, value))[1] != value:
                    previous_key, previous_value = already_set[input_id]
                    raise ValueError(
                        f"Can not set the position ids for '{key}' to {value} because it was already "
                        f"set to {previous_value} by key '{previous_key}'. Note that both, '{key}' and "
                        f"'{previous_key}', have the same id ({input_id}), so their position_ids need to "
                        f"be also the same (position id mapping: {mapping})."
                    )
                position_ids[input_ids.eq(input_id)] = value
                already_set[input_id] = key, value
            return position_ids
        else:
            raise ValueError(
                f"decoder_position_id_mode={self.decoder_position_id_mode} not supported!"
            )

    def prepare_decoder_inputs(
        self,
        input_ids: torch.LongTensor,
        encoder_input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Dict[str, torch.Tensor]:
        inputs: Dict[str, torch.Tensor] = {}
        if self.use_prepared_position_ids:
            if position_ids is None:
                position_ids = self.prepare_decoder_position_ids(input_ids=input_ids)
            inputs["position_ids"] = position_ids

        inputs["input_ids"] = self.prepare_decoder_input_ids(
            input_ids=input_ids,
            encoder_input_ids=encoder_input_ids,
        )
        return inputs

    def forward(
        self,
        last_hidden_state,
        encoder_input_ids,
        encoder_last_hidden_state,
        encoder_attention_mask,
        labels: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        constraints: Optional[torch.LongTensor] = None,
    ):
        # assemble the logits
        logits = last_hidden_state.new_full(
            (
                last_hidden_state.size(0),
                last_hidden_state.size(1),
                self.pointer_offset + encoder_input_ids.size(-1),
            ),
            fill_value=-1e24,
        )

        # eos and label scores depend only on the decoder output
        # bsz x max_len x 1
        eos_scores = F.linear(last_hidden_state, self.embeddings.weight[[self.eos_token_id]])
        label_embeddings = self.embeddings.weight[self.label_token_ids]
        # bsz x max_len x num_class
        label_scores = F.linear(last_hidden_state, label_embeddings)

        # the pointer depends on the src token embeddings, the encoder output and the decoder output
        # bsz x max_bpe_len x hidden_size
        src_outputs = encoder_last_hidden_state
        if getattr(self, "encoder_mlp", None) is not None:
            src_outputs = self.encoder_mlp(src_outputs)

        # bsz x max_word_len x hidden_size
        input_embed = self.embeddings(encoder_input_ids)

        # bsz x max_len x max_word_len
        word_scores = torch.einsum("blh,bnh->bln", last_hidden_state, src_outputs)
        gen_scores = torch.einsum("blh,bnh->bln", last_hidden_state, input_embed)
        avg_word_scores = (gen_scores + word_scores) / 2

        # never point to the padding or the eos token in the encoder input
        # TODO: why not excluding the bos token? seems to give worse results, but not tested extensively
        mask_invalid = encoder_attention_mask.eq(0) | encoder_input_ids.eq(self.eos_token_id)
        avg_word_scores = avg_word_scores.masked_fill(mask_invalid.unsqueeze(1), -1e32)

        # Note: the remaining row in logits contains the score for the bos token which should be never generated!
        logits[:, :, [self.eos_id]] = eos_scores
        logits[:, :, self.label_ids] = label_scores
        logits[:, :, self.pointer_offset :] = avg_word_scores

        loss = None
        # compute the loss if labels are provided
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            logits_resized = logits.reshape(-1, logits.size(-1))
            labels_resized = labels.reshape(-1)
            if decoder_attention_mask is None:
                raise ValueError("decoder_attention_mask must be provided to compute the loss!")
            mask_resized = decoder_attention_mask.reshape(-1)
            labels_masked = labels_resized.masked_fill(
                ~mask_resized.to(torch.bool), loss_fct.ignore_index
            )
            loss = loss_fct(logits_resized, labels_masked)

        # compute the constraints loss if constraints are provided
        if constraints is not None:
            if getattr(self, "constraints_encoder_mlp", None) is not None:
                # TODO: is it fine to apply constraints_encoder_mlp to both src_outputs and label_embeddings?
                #  This is what the original code seems to do, but this is different from the usage of encoder_mlp.
                constraints_src_outputs = self.constraints_encoder_mlp(src_outputs)
                constraints_label_embeddings = self.constraints_encoder_mlp(label_embeddings)
            else:
                constraints_src_outputs = src_outputs
                constraints_label_embeddings = label_embeddings
            constraints_label_scores = F.linear(last_hidden_state, constraints_label_embeddings)
            # bsz x max_len x max_word_len
            constraints_word_scores = torch.einsum(
                "blh,bnh->bln", last_hidden_state, constraints_src_outputs
            )
            constraints_logits = last_hidden_state.new_full(
                (
                    last_hidden_state.size(0),
                    last_hidden_state.size(1),
                    self.pointer_offset + encoder_input_ids.size(-1),
                ),
                fill_value=-1e24,
            )
            constraints_logits[:, :, self.label_ids] = constraints_label_scores
            constraints_logits[:, :, self.pointer_offset :] = constraints_word_scores

            mask = constraints >= 0
            constraints_logits_valid = constraints_logits[mask]
            constraints_valid = constraints[mask]
            loss_c = F.binary_cross_entropy(
                torch.sigmoid(constraints_logits_valid), constraints_valid.float()
            )

            if loss is None:
                loss = loss_c
            else:
                loss += loss_c

        return logits, loss
