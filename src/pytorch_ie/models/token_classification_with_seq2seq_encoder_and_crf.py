import logging
from typing import Any, Dict, MutableMapping, Optional, Tuple, Union

import torch
from pie_core import Model
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import FloatTensor, LongTensor, nn
from transformers import AutoConfig, AutoModel, BatchEncoding, get_linear_schedule_with_warmup
from transformers.modeling_outputs import TokenClassifierOutput
from typing_extensions import TypeAlias

from pytorch_ie.models.interface import RequiresModelNameOrPath, RequiresNumClasses

from .common import ModelWithBoilerplate
from .components.seq2seq_encoder import build_seq2seq_encoder

# model inputs / outputs / targets
InputType: TypeAlias = BatchEncoding
OutputType: TypeAlias = TokenClassifierOutput
TargetType: TypeAlias = MutableMapping[str, Union[LongTensor, FloatTensor]]
# step inputs (batch) / outputs (loss)
StepInputType: TypeAlias = Tuple[InputType, TargetType]
StepOutputType: TypeAlias = FloatTensor

HF_MODEL_TYPE_TO_CLASSIFIER_DROPOUT_ATTRIBUTE = {
    "bert": "hidden_dropout_prob",
    "roberta": "hidden_dropout_prob",
    "albert": "classifier_dropout_prob",
    "distilbert": "seq_classif_dropout",
    "deberta-v2": "hidden_dropout_prob",
    "longformer": "hidden_dropout_prob",
}

logger = logging.getLogger(__name__)


@Model.register()
class TokenClassificationModelWithSeq2SeqEncoderAndCrf(
    ModelWithBoilerplate[InputType, OutputType, TargetType, StepOutputType],
    RequiresNumClasses,
    RequiresModelNameOrPath,
):
    """A token classification model that wraps a (pretrained) model loaded with AutoModel from the
    transformers library. The model can optionally be followed by a seq2seq encoder (e.g. an LSTM).
    Finally, Conditional Random Fields (CRFs) can be used to decode the predictions.

    The model is trained with a cross-entropy loss function and uses the Adam optimizer.

    Note that for training, the labels for the special tokens (as well as for padding tokens)
    are expected to have the value label_pad_id (-100 by default, which is the default ignore_index
    value for the CrossEntropyLoss). The predictions for these tokens are also replaced with
    label_pad_id to match the training labels for correct metric calculation. Therefore, the model
    requires the special_tokens_mask and attention_mask (for padding) to be passed as inputs.

    Args:
        model_name_or_path: The name or path of the (pretrained) transformer model to use.
        num_classes: The number of classes to predict.
        learning_rate: The learning rate to use for training.
        task_learning_rate: The learning rate to use for the task-specific parameters, i.e.
            for the sequence-to-sequence encoder, classification head, and CRF. If None, the
            learning_rate is used for all parameters.
        use_crf: Whether to use a CRF to decode the predictions.
        label_pad_id: The label id to use for padding labels (at the padding token positions
            as well as for the special tokens).
        special_token_label_id: The label id to use for special tokens (e.g. [CLS], [SEP]). This
            is used to replace the targets for special tokens with the label_pad_id before passing
            them to the CRF because the CRF does not allow the first token to be masked out.
        classifier_dropout: The dropout probability to use for the classification head.
        freeze_base_model: Whether to freeze the base model (i.e. the transformer) during training.
        warmup_proportion: The proportion of training steps to use for the linear warmup.
        seq2seq_encoder: A dictionary with the configuration for the seq2seq encoder. If None, no
            seq2seq encoder is used. See ./components/seq2seq_encoder.py for further information.
    """

    def __init__(
        self,
        model_name_or_path: str,
        num_classes: int,
        learning_rate: float = 1e-5,
        task_learning_rate: Optional[float] = None,
        use_crf: bool = True,
        label_pad_id: int = -100,
        special_token_label_id: int = 0,
        classifier_dropout: Optional[float] = None,
        freeze_base_model: bool = False,
        warmup_proportion: float = 0.1,
        seq2seq_encoder: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.special_token_label_id = special_token_label_id

        self.learning_rate = learning_rate
        self.warmup_proportion = warmup_proportion
        self.task_learning_rate = task_learning_rate
        self.label_pad_id = label_pad_id
        self.num_classes = num_classes

        config = AutoConfig.from_pretrained(model_name_or_path)
        if self.is_from_pretrained:
            self.model = AutoModel.from_config(config=config)
        else:
            self.model = AutoModel.from_pretrained(model_name_or_path, config=config)

        if freeze_base_model:
            self.model.requires_grad_(False)

        hidden_size = config.hidden_size
        self.seq2seq_encoder = None
        if seq2seq_encoder is not None:
            self.seq2seq_encoder, hidden_size = build_seq2seq_encoder(
                config=seq2seq_encoder, input_size=hidden_size
            )

        if classifier_dropout is None:
            # Get the classifier dropout value from the Huggingface model config.
            # This is a bit of a mess since some Configs use different variable names or change the semantics
            # of the dropout (e.g. DistilBert has one dropout prob for QA and one for Seq classification, and a
            # general one for embeddings, encoder and pooler).
            classifier_dropout_attr = HF_MODEL_TYPE_TO_CLASSIFIER_DROPOUT_ATTRIBUTE.get(
                config.model_type, "classifier_dropout"
            )
            if hasattr(config, classifier_dropout_attr):
                classifier_dropout = getattr(config, classifier_dropout_attr)
            else:
                raise ValueError(
                    f"The config {type(config),__name__} loaded from {model_name_or_path} has no attribute "
                    f"{classifier_dropout_attr}"
                )
        self.dropout = nn.Dropout(classifier_dropout)

        self.classifier = nn.Linear(hidden_size, num_classes)

        if use_crf:
            try:
                from torchcrf import CRF
            except ImportError:
                raise ImportError(
                    "To use CRFs, the torchcrf package must be installed. "
                    "You can install it with `pip install pytorch-crf`."
                )

            self.crf = CRF(num_tags=num_classes, batch_first=True)
        else:
            self.crf = None

    def decode(self, inputs: InputType, outputs: OutputType) -> TargetType:
        result: MutableMapping[str, Union[LongTensor, FloatTensor]] = {}
        logits = outputs.logits
        attention_mask = inputs["attention_mask"]
        special_tokens_mask = inputs["special_tokens_mask"]
        attention_mask_bool = attention_mask.to(torch.bool)
        if self.crf is not None:
            decoded_tags = self.crf.decode(emissions=logits, mask=attention_mask_bool)
            # pad the decoded tags to the length of the logits to have the same shape as when not using the crf
            seq_len = logits.shape[1]
            padded_tags = [
                tags + [self.label_pad_id] * (seq_len - len(tags)) for tags in decoded_tags
            ]
            tags_tensor = torch.tensor(padded_tags, device=logits.device).to(torch.long)
        else:
            # get the max index for each token from the logits
            tags_tensor = torch.argmax(logits, dim=-1).to(torch.long)
        # set the padding and special tokens to the label_pad_id
        mask = attention_mask_bool & ~special_tokens_mask.to(torch.bool)
        tags_tensor = tags_tensor.masked_fill(~mask, self.label_pad_id)
        probs = torch.softmax(logits, dim=-1)

        assert isinstance(tags_tensor, LongTensor)
        assert isinstance(probs, FloatTensor)
        result["labels"] = tags_tensor
        # TODO: is it correct to use this also in the case of the crf?
        result["probabilities"] = probs

        return result

    def forward(
        self, inputs: InputType, targets: Optional[TargetType] = None
    ) -> TokenClassifierOutput:
        inputs_without_special_tokens_mask = {
            k: v for k, v in inputs.items() if k != "special_tokens_mask"
        }
        outputs = self.model(**inputs_without_special_tokens_mask)
        sequence_output = outputs[0]

        if self.seq2seq_encoder is not None:
            sequence_output = self.seq2seq_encoder(sequence_output)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if targets is not None:
            labels = targets["labels"]
            if self.crf is not None:
                # Overwrite the padding labels with ignore_index. Note that this is different from the
                # attention_mask, because the attention_mask includes special tokens, whereas the labels
                # are set to label_pad_id also for special tokens (e.g. [CLS]). We need handle all
                # occurrences of label_pad_id because usually that index is out of range with respect to
                # the number of logits in which case the crf would complain. However, we can not simply
                # pass a mask to the crf that also masks out the special tokens, because the crf does not
                # allow the first token to be masked out.
                mask_pad_or_special = labels == self.label_pad_id
                labels_valid = labels.masked_fill(mask_pad_or_special, self.special_token_label_id)
                # the crf expects a bool mask
                if "attention_mask" in inputs:
                    mask_bool = inputs["attention_mask"].to(torch.bool)
                else:
                    mask_bool = None
                log_likelihood = self.crf(emissions=logits, tags=labels_valid, mask=mask_bool)
                loss = -log_likelihood
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=self.label_pad_id)
                loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        if self.task_learning_rate is not None:
            all_params = dict(self.named_parameters())
            base_model_params = dict(self.model.named_parameters(prefix="model"))
            task_params = {k: v for k, v in all_params.items() if k not in base_model_params}
            optimizer = torch.optim.AdamW(
                [
                    {"params": base_model_params.values(), "lr": self.learning_rate},
                    {"params": task_params.values(), "lr": self.task_learning_rate},
                ]
            )
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        if self.warmup_proportion > 0.0:
            stepping_batches = self.trainer.estimated_stepping_batches
            scheduler = get_linear_schedule_with_warmup(
                optimizer, int(stepping_batches * self.warmup_proportion), stepping_batches
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        else:
            return optimizer
