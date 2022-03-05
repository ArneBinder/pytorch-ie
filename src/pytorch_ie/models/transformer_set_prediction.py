import collections
from typing import MutableMapping, Any, Dict, Tuple, Optional, List

import pytorch_lightning as pl

from pytorch_ie import Document
from pytorch_ie.data import Metadata
from torch import Tensor
from transformers import AdamW, AutoConfig, AutoModel, BertConfig, get_linear_schedule_with_warmup

from pytorch_ie.metrics.set_fbeta import SetFbetaScore
from pytorch_ie.models.set_prediction.loss.loss_functions import (
    BinaryCrossEntropyLossFunction,
    CrossEntropyLossFunction,
)
from pytorch_ie.models.set_prediction.loss.set_criterion import SetCriterion
from pytorch_ie.models.set_prediction.matching.cost_functions import (
    BinaryCrossEntropyCostFunction,
    CrossEntropyCostFunction,
)
from pytorch_ie.models.set_prediction.matching.matcher import HungarianMatcher
from pytorch_ie.models.set_prediction.set_decoder import (
    SpanLabelAndMaskDecoder,
    SpanLabelDecoder,
    SpanLabelJointDecoder,
)
from pytorch_ie.models.set_prediction.set_transformer import SetTransformer


TransformerSetPredictionModelBatchEncoding = MutableMapping[str, Any]
TransformerSetPredictionModelBatchOutput = Dict[str, Any]

TransformerSetPredictionModelStepBatchEncoding = Tuple[
    Dict[str, Tensor],
    Optional[Dict[str, Dict[str, List[Tensor]]]],
    List[Metadata],
    List[Document],
]


def flatten_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class TransformerSetPredictionModel(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_classes: int,
        t_total: int,
        num_queries: int = 25,
        none_coef: float = 0.1,
        decoder_num_hidden_layers: int = 6,
        learning_rate: float = 1e-4,
        encoder_learning_rate: float = 5e-5,
        warmup_proportion: float = 0.1,
        # ignore_index: int = 0,
        freeze_encoder: bool = False,
        weight_decay: float = 0.2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.t_total = t_total
        self.learning_rate = learning_rate
        self.encoder_learning_rate = encoder_learning_rate
        self.warmup_proportion = warmup_proportion
        self.weight_decay = weight_decay

        config = AutoConfig.from_pretrained(model_name_or_path)
        self.encoder = AutoModel.from_pretrained(model_name_or_path, config=config)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        decoder_config = BertConfig(
            is_decoder=True,
            hidden_size=self.encoder.config.hidden_size,
            num_hidden_layers=decoder_num_hidden_layers,
            num_attention_heads=self.encoder.config.num_attention_heads,
            add_cross_attention=True,
        )

        set_decoders = [
            (
                "entities",
                # SpanLabelDecoder(
                SpanLabelJointDecoder(
                    config=decoder_config,
                    num_queries=num_queries,
                    num_labels=num_classes,
                    query_prototypes="learned",
                ),
            ),
        ]

        matchers = {
            "entities": HungarianMatcher(
                cost_functions={
                    "label_ids": CrossEntropyCostFunction(),
                    "start_index": CrossEntropyCostFunction(),
                    "end_index": CrossEntropyCostFunction(),
                    # "span_position": SpanPositionCostFunction(),
                    "span_mask": BinaryCrossEntropyCostFunction(),
                },
                cost_weights={
                    "label_ids": 1.0,
                    "start_index": 1.0,
                    "end_index": 1.0,
                    # "span_position": 1.0,
                    "span_mask": 1.0,
                },
            ),
        }

        set_criteria = {
            "entities": SetCriterion(
                loss_functions={
                    "label_ids": CrossEntropyLossFunction(
                        num_classes=num_classes,
                        none_index=num_classes,
                        none_weight=none_coef,
                    ),
                    "start_index": CrossEntropyLossFunction(),
                    "end_index": CrossEntropyLossFunction(),
                    # "span_position": SpanPositionLossFunction(),
                    "span_mask": BinaryCrossEntropyLossFunction(),
                },
                loss_weights={
                    "label_ids": 1.0,
                    "start_index": 1.0,
                    "end_index": 1.0,
                    # "span_position": 1.0,
                    "span_mask": 1.0,
                },
            ),
        }

        self.model = SetTransformer(
            encoder=self.encoder,
            set_decoders=set_decoders,
            matchers=matchers,
            set_criteria=set_criteria,
        )

        self.train_f1 = SetFbetaScore(none_index=num_classes, beta=2.0)
        self.val_f1 = SetFbetaScore(none_index=num_classes, beta=2.0)

    def forward(self, input_: TransformerSetPredictionModelBatchEncoding) -> TransformerSetPredictionModelBatchOutput:  # type: ignore
        output = self.model(**input_)
        return output

    def training_step(self, batch: TransformerSetPredictionModelStepBatchEncoding, batch_idx):
        input_, target, _, docs = batch

        output = self(input_)

        losses = self.model.get_loss(output, target)

        flat_losses = flatten_dict(losses)
        for loss_name, loss in flat_losses.items():
            self.log("train/" + loss_name, loss)

        total_loss = sum([loss for _, loss in flat_losses.items()])
        self.log("train/loss", total_loss, prog_bar=True)

        self.train_f1(output["entities"], target["entities"])
        self.log("train/entities_f1", self.train_f1, on_step=True, on_epoch=False, prog_bar=True)

        return total_loss

    def validation_step(self, batch: TransformerSetPredictionModelStepBatchEncoding, batch_idx):
        input_, target, _, docs = batch

        output = self(input_)

        losses = self.model.get_loss(output, target)

        flat_losses = flatten_dict(losses)
        for loss_name, loss in flat_losses.items():
            self.log("val/" + loss_name, loss)

        total_loss = sum([loss for _, loss in flat_losses.items()])
        self.log("val/loss", total_loss, prog_bar=True)

        self.val_f1(output["entities"], target["entities"])
        self.log("val/entities_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)

        return total_loss

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay) and not n.startswith("model.encoder")
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay) and not n.startswith("model.encoder")
                ],
                "weight_decay": 0.0,
            },
            # The following groups handle the optimization of the encoder (which is typically pretrained and may require special handling)
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay) and n.startswith("model.encoder")
                ],
                "weight_decay": self.weight_decay,
                "lr": self.encoder_learning_rate,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay) and n.startswith("model.encoder")
                ],
                "weight_decay": 0.0,
                "lr": self.encoder_learning_rate,
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        scheduler = {
            "scheduler": get_linear_schedule_with_warmup(
                optimizer, int(self.t_total * self.warmup_proportion), self.t_total
            ),
            "interval": "step"
        }
        return [optimizer], [scheduler]
