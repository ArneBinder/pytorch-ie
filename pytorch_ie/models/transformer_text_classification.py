from typing import Any, Dict, List, MutableMapping, Optional, Tuple

import torchmetrics
from torch import Tensor, nn
from transformers import AdamW, AutoConfig, AutoModel, get_linear_schedule_with_warmup

from pytorch_ie.core.pytorch_ie import PyTorchIEModel

TransformerTextClassificationModelBatchEncoding = MutableMapping[str, Any]
TransformerTextClassificationModelBatchOutput = Dict[str, Any]

TransformerTextClassificationModelStepBatchEncoding = Tuple[
    Dict[str, Tensor],
    Optional[Tensor],
]


class TransformerTextClassificationModel(PyTorchIEModel):
    def __init__(
        self,
        model_name_or_path: str,
        t_total: int,
        num_classes: int,
        tokenizer_vocab_size: int,
        ignore_index: Optional[int] = None,
        learning_rate: float = 1e-5,
        task_learning_rate: float = 1e-4,
        warmup_proportion: float = 0.1,
        freeze_model: bool = False,
        multi_label: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.t_total = t_total
        self.learning_rate = learning_rate
        self.task_learning_rate = task_learning_rate
        self.warmup_proportion = warmup_proportion

        config = AutoConfig.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path, config=config)
        self.model.resize_token_embeddings(tokenizer_vocab_size)

        # if freeze_model:
        #     for param in self.model.parameters():
        #         param.requires_grad = False

        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        self.classifier = nn.Linear(config.hidden_size, num_classes)

        self.loss_fct = nn.BCEWithLogitsLoss() if multi_label else nn.CrossEntropyLoss()

        self.train_f1 = torchmetrics.F1(num_classes=num_classes, ignore_index=ignore_index)
        self.val_f1 = torchmetrics.F1(num_classes=num_classes, ignore_index=ignore_index)

    def forward(self, input_: TransformerTextClassificationModelBatchEncoding) -> TransformerTextClassificationModelBatchOutput:  # type: ignore
        output = self.model(**input_)

        hidden_state = output.last_hidden_state

        cls_embeddings = hidden_state[:, 0, :]
        logits = self.classifier(cls_embeddings)

        return {"logits": logits}

    def training_step(self, batch: TransformerTextClassificationModelStepBatchEncoding, batch_idx):  # type: ignore
        input_, target = batch
        assert target is not None, "target has to be available for training"

        logits = self(input_)["logits"]

        loss = self.loss_fct(logits, target)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        self.train_f1(logits, target)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: TransformerTextClassificationModelStepBatchEncoding, batch_idx):  # type: ignore
        input_, target = batch
        assert target is not None, "target has to be available for validation"

        logits = self(input_)["logits"]

        loss = self.loss_fct(logits, target)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        self.val_f1(logits, target)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(self.t_total * self.warmup_proportion), self.t_total
        )
        return [optimizer], [scheduler]

        # param_optimizer = list(self.named_parameters())
        # # TODO: this needs fixing (does not work models other than BERT)
        # optimizer_grouped_parameters = [
        #     {"params": [p for n, p in param_optimizer if "bert" in n]},
        #     {
        #         "params": [p for n, p in param_optimizer if "bert" not in n],
        #         "lr": self.task_learning_rate,
        #     },
        # ]
        # optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer, int(self.t_total * self.warmup_proportion), self.t_total
        # )
        # return [optimizer], [scheduler]
        # return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
