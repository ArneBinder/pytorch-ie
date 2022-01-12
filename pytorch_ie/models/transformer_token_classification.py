from typing import Any, Dict, List, Optional, Tuple

import torch
import torchmetrics
from torch import Tensor
from transformers import AutoConfig, AutoModelForTokenClassification, BatchEncoding

from pytorch_ie.core.pytorch_ie import PyTorchIEModel
from pytorch_ie.data import Document, Metadata

TransformerTokenClassificationInputEncoding = BatchEncoding
TransformerTokenClassificationTargetEncoding = List[int]

TransformerTokenClassificationModelBatchEncoding = BatchEncoding
TransformerTokenClassificationModelBatchOutput = Dict[str, Any]

TransformerTokenClassificationModelStepBatchEncoding = Tuple[
    Dict[str, Tensor],
    Tensor,
    List[Metadata],
    List[Document],
]


class TransformerTokenClassificationModel(PyTorchIEModel):
    def __init__(
        self,
        model_name_or_path: str,
        num_classes: int,
        learning_rate: float = 1e-5,
        label_pad_token_id: int = -100,
        ignore_index: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.label_pad_token_id = label_pad_token_id

        config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_classes)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name_or_path, config=config
        )

        self.train_f1 = torchmetrics.F1(num_classes=num_classes, ignore_index=ignore_index)
        self.val_f1 = torchmetrics.F1(num_classes=num_classes, ignore_index=ignore_index)

    def forward(
        self, input_: TransformerTokenClassificationModelBatchEncoding
    ) -> TransformerTokenClassificationModelBatchOutput:
        return self.model(**input_)

    def training_step(
        self, batch: TransformerTokenClassificationModelStepBatchEncoding, batch_idx
    ):
        input_, target, _, _ = batch

        input_["labels"] = target
        output = self(input_)

        loss = output.loss
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        target_flat = target.view(-1)

        valid_indices = target_flat != self.label_pad_token_id
        valid_logits = output.logits.view(-1, self.hparams.num_classes)[valid_indices]
        valid_target = target_flat[valid_indices]

        self.train_f1(valid_logits, valid_target)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(
        self, batch: TransformerTokenClassificationModelStepBatchEncoding, batch_idx
    ):
        input_, target, _, _ = batch

        input_["labels"] = target
        output = self(input_)

        loss = output.loss
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        target_flat = target.view(-1)

        valid_indices = target_flat != self.label_pad_token_id
        valid_logits = output.logits.view(-1, self.hparams.num_classes)[valid_indices]
        valid_target = target_flat[valid_indices]

        self.val_f1(valid_logits, valid_target)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
