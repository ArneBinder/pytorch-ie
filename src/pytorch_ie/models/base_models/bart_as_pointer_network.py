from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch.utils.checkpoint
from torch import nn
from torch.nn import Parameter
from torch.optim import Optimizer
from transformers import BartConfig, BartModel, BartPreTrainedModel, GenerationConfig
from transformers.modeling_outputs import Seq2SeqLMOutput, Seq2SeqModelOutput
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers.utils import logging

from ..components.pointer_head import PointerHead
from .bart_with_decoder_position_ids import BartModelWithDecoderPositionIds

logger = logging.get_logger(__name__)


def get_layer_norm_parameters(
    named_parameters: Iterator[Tuple[str, Parameter]],
) -> Iterator[Parameter]:
    return (
        param for name, param in named_parameters if "layernorm" in name or "layer_norm" in name
    )


def get_non_layer_norm_parameters(
    named_parameters: Iterator[Tuple[str, Parameter]],
) -> Iterator[Parameter]:
    return (
        param
        for name, param in named_parameters
        if not ("layernorm" in name or "layer_norm" in name)
    )


class BartAsPointerNetworkConfig(BartConfig):
    def __init__(
        self,
        # respective token ids for the label-, eos-, and pad ids. Can be used as a mapping from the
        # target ids to the token ids.
        target_token_ids: Optional[List[int]] = None,
        # token id mapping to better initialize the label embedding weights
        embedding_weight_mapping: Optional[Dict[Union[int, str], List[int]]] = None,
        # special decoder position id handling
        decoder_position_id_mode: Optional[str] = None,
        decoder_position_id_pattern: Optional[List[int]] = None,
        decoder_position_id_mapping: Optional[Dict[int, int]] = None,
        # other parameters
        use_encoder_mlp: bool = True,
        use_constraints_encoder_mlp: bool = False,
        # optimizer
        lr: float = 5e-5,
        task_lr: Optional[float] = None,
        weight_decay: float = 1e-2,
        head_decay: Optional[float] = None,
        shared_decay: Optional[float] = None,
        encoder_layer_norm_decay: Optional[float] = 0.001,
        decoder_layer_norm_decay: Optional[float] = None,
        # other BartConfig parameters
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.target_token_ids = target_token_ids

        self.embedding_weight_mapping = embedding_weight_mapping

        self.use_encoder_mlp = use_encoder_mlp
        self.use_constraints_encoder_mlp = use_constraints_encoder_mlp

        self.decoder_position_id_mode = decoder_position_id_mode
        self.decoder_position_id_pattern = decoder_position_id_pattern
        self.decoder_position_id_mapping = decoder_position_id_mapping

        self.lr = lr
        self.task_lr = task_lr
        self.weight_decay = weight_decay
        self.head_decay = head_decay
        self.shared_decay = shared_decay
        self.encoder_layer_norm_decay = encoder_layer_norm_decay
        self.decoder_layer_norm_decay = decoder_layer_norm_decay


class BartAsPointerNetwork(BartPreTrainedModel):
    config_class = BartAsPointerNetworkConfig
    base_model_prefix = "model"
    _tied_weights_keys = [
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
    ]

    def __init__(self, config: BartAsPointerNetworkConfig):
        super().__init__(config)
        if self.config.decoder_position_id_mode is not None:
            self.model = BartModelWithDecoderPositionIds(config)
        else:
            self.model = BartModel(config)

        self.pointer_head = PointerHead(
            # target space ids
            bos_id=self.model.config.bos_token_id,
            eos_id=self.model.config.eos_token_id,
            pad_id=self.model.config.pad_token_id,
            # decoder-input token ids
            target_token_ids=self.model.config.target_token_ids,
            # embeddings
            embeddings=self.model.decoder.embed_tokens,
            embedding_weight_mapping=self.model.config.embedding_weight_mapping,
            # other parameters
            use_encoder_mlp=self.model.config.use_encoder_mlp,
            use_constraints_encoder_mlp=self.model.config.use_constraints_encoder_mlp,
            decoder_position_id_mode=self.model.config.decoder_position_id_mode,
            decoder_position_id_pattern=self.model.config.decoder_position_id_pattern,
            decoder_position_id_mapping=self.model.config.decoder_position_id_mapping,
        )

        # Initialize weights and apply final processing
        self.post_init()

    @classmethod
    def _load_pretrained_model(
        cls,
        *args,
        **kwargs,
    ):
        (
            model,
            missing_keys,
            unexpected_keys,
            mismatched_keys,
            offload_index,
            error_msgs,
        ) = super()._load_pretrained_model(*args, **kwargs)
        # adjust the model after loading the original model (e.g. vanilla BartModel)
        model.adjust_after_loading_original_model()
        return model, missing_keys, unexpected_keys, mismatched_keys, offload_index, error_msgs

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # we also need to update the embeddings in the pointer head
        self.pointer_head.set_embeddings(new_embeddings)
        return new_embeddings

    def adjust_after_loading_original_model(self):
        # target_token_ids contains all new target tokens for the labels and new tokens were added to the end
        # of the vocabulary, so we can use its maximum to resize the embedding weights
        self.resize_token_embeddings(new_num_tokens=max(self.config.target_token_ids) + 1)
        # initialize the newly added embeddings for the labels with better weights from the original embeddings
        self.pointer_head.overwrite_embeddings_with_mapping()

        # adjust generation settings
        # set the correct decoder_start_token_id
        self.config.decoder_start_token_id = self.config.bos_token_id
        # disable ForcedBOSTokenLogitsProcessor
        self.config.forced_bos_token_id = None
        # disable ForcedEOSTokenLogitsProcessor
        self.config.forced_eos_token_id = None
        # update the generation config accordingly
        self.generation_config = GenerationConfig.from_model_config(self.config)

    def base_model_named_params(self, prefix: str = "") -> Iterator[Tuple[str, Parameter]]:
        yield from self.model.named_parameters(prefix=prefix + self.base_model_prefix)

    def head_named_params(self, prefix: str = "") -> Iterator[Tuple[str, Parameter]]:
        base_model_param_names = {
            name for name, param in self.base_model_named_params(prefix=prefix)
        }
        for name, param in self.named_parameters(prefix=prefix):
            if name not in base_model_param_names:
                yield name, param

    def encoder_only_named_params(self, prefix: str = "") -> Iterator[Tuple[str, Parameter]]:
        shared_params = set(dict(self.encoder_decoder_shared_named_params(prefix=prefix)).values())
        for name, param in self.model.encoder.named_parameters(
            prefix=prefix + self.base_model_prefix + ".encoder"
        ):
            if param not in shared_params:
                yield name, param

    def decoder_only_named_params(self, prefix: str = "") -> Iterator[Tuple[str, Parameter]]:
        shared_params = set(dict(self.encoder_decoder_shared_named_params(prefix=prefix)).values())
        for name, param in self.model.decoder.named_parameters(
            prefix=prefix + self.base_model_prefix + ".decoder"
        ):
            if param not in shared_params:
                yield name, param

    def encoder_decoder_shared_named_params(
        self, prefix: str = ""
    ) -> Iterator[Tuple[str, Parameter]]:
        encoder_params = set(self.model.encoder.parameters())
        decoder_params = set(self.model.decoder.parameters())
        for name, param in self.base_model_named_params(prefix=prefix):
            if param in encoder_params and param in decoder_params:
                yield name, param

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    # @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    # @add_end_docstrings(BART_GENERATION_EXAMPLE)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        decoder_position_ids: Optional[torch.LongTensor] = None,
        constraints: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""Labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*): Labels
        for computing the masked language modeling loss. Indices should either be in `[0, ...,
        config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100`
        are ignored (masked), the loss is only computed for the tokens with labels in `[0, ...,
        config.vocab_size]`.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning(
                    "The `use_cache` argument is changed to `False` since `labels` is provided."
                )
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        if decoder_input_ids is None:
            # we can not create the decoder_input_ids from input_ids, because we need the
            # encoder_input_ids for the pointer network
            raise ValueError("decoder_input_ids has to be set!")

        if input_ids is None:
            raise ValueError("input_ids has to be set!")

        # this adjusts the input_ids and, if available, the position_ids
        decoder_inputs = self.pointer_head.prepare_decoder_inputs(
            input_ids=decoder_input_ids,
            # in the case of generation (with past_key_values) the position_ids are already prepared
            position_ids=decoder_position_ids,
            encoder_input_ids=input_ids,
        )

        model_inputs = dict(
            input_ids=input_ids,
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        for k, v in decoder_inputs.items():
            model_inputs[f"decoder_{k}"] = v
        outputs = self.model(**model_inputs)

        if not isinstance(outputs, Seq2SeqModelOutput):
            raise ValueError(
                "Inconsistent output: The output of the model forward should be of type "
                f"`Seq2SeqLMOutput`, but is of type `{type(outputs)}`."
            )
        logits, loss = self.pointer_head(
            last_hidden_state=outputs.last_hidden_state,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_input_ids=input_ids,
            encoder_attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
            constraints=constraints,
        )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        encoder_input_ids,  # added for pointer network
        encoder_attention_mask,  # added for pointer network
        past_key_values=None,
        attention_mask=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        result = {}
        if self.pointer_head.use_prepared_position_ids:
            # we need to prepare the position ids for the decoder here, because later we do not have the full
            # input_ids anymore
            result["decoder_position_ids"] = self.pointer_head.prepare_decoder_position_ids(
                input_ids=decoder_input_ids
            )

        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

            if "decoder_position_ids" in result:
                result["decoder_position_ids"] = result["decoder_position_ids"][
                    :, remove_prefix_length:
                ]

        result.update(
            {
                "input_ids": encoder_input_ids,
                "encoder_outputs": encoder_outputs,
                "past_key_values": past_key_values,
                "decoder_input_ids": decoder_input_ids,
                "attention_mask": encoder_attention_mask,
                "decoder_attention_mask": decoder_attention_mask,
                "head_mask": head_mask,
                "decoder_head_mask": decoder_head_mask,
                "cross_attn_head_mask": cross_attn_head_mask,
                "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            }
        )
        return result

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(
            labels, self.config.pad_token_id, self.config.decoder_start_token_id
        )

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past[:2]
                )
                + layer_past[2:],
            )
        return reordered_past

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:
        result = super()._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor=inputs_tensor,
            model_kwargs=model_kwargs,
            model_input_name=model_input_name,
        )
        # add items that are needed for pointer network
        result["encoder_input_ids"] = inputs_tensor
        result["encoder_attention_mask"] = result["attention_mask"]
        return result

    def configure_optimizer(self) -> Optimizer:
        parameters = []

        # head parameters
        head_decay = (
            self.config.head_decay
            if self.config.head_decay is not None
            else self.config.weight_decay
        )
        params = {
            "lr": self.config.task_lr if self.config.task_lr is not None else self.config.lr,
            "weight_decay": head_decay,
            "params": dict(self.head_named_params()).values(),
        }
        parameters.append(params)

        # decoder only layer norm parameters
        decoder_layer_norm_decay = (
            self.config.decoder_layer_norm_decay
            if self.config.decoder_layer_norm_decay is not None
            else self.config.weight_decay
        )
        params = {
            "lr": self.config.lr,
            "weight_decay": decoder_layer_norm_decay,
            "params": get_layer_norm_parameters(self.decoder_only_named_params()),
        }
        parameters.append(params)

        # decoder only other parameters
        params = {
            "lr": self.config.lr,
            "weight_decay": self.config.weight_decay,
            "params": get_non_layer_norm_parameters(self.decoder_only_named_params()),
        }
        parameters.append(params)

        # encoder only layer norm parameters
        encoder_layer_norm_decay = (
            self.config.encoder_layer_norm_decay
            if self.config.encoder_layer_norm_decay is not None
            else self.config.weight_decay
        )
        params = {
            "lr": self.config.lr,
            "weight_decay": encoder_layer_norm_decay,
            "params": get_layer_norm_parameters(self.encoder_only_named_params()),
        }
        parameters.append(params)

        # encoder only other parameters
        params = {
            "lr": self.config.lr,
            "weight_decay": self.config.weight_decay,
            "params": get_non_layer_norm_parameters(self.encoder_only_named_params()),
        }
        parameters.append(params)

        # encoder-decoder shared parameters
        shared_decay = (
            self.config.shared_decay
            if self.config.shared_decay is not None
            else self.config.weight_decay
        )
        params = {
            "lr": self.config.lr,
            "weight_decay": shared_decay,
            "params": dict(self.encoder_decoder_shared_named_params()).values(),
        }
        parameters.append(params)

        return torch.optim.AdamW(parameters)
