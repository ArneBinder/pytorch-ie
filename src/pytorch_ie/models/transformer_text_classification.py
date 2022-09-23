from typing import Any, Dict, MutableMapping, Optional, Tuple

import torchmetrics
from torch import Tensor, nn
from transformers import AdamW, AutoConfig, AutoModel, get_linear_schedule_with_warmup

from pytorch_ie.core import PyTorchIEModel

TransformerTextClassificationModelBatchEncoding = MutableMapping[str, Any]
TransformerTextClassificationModelBatchOutput = Dict[str, Any]

TransformerTextClassificationModelStepBatchEncoding = Tuple[
    Dict[str, Tensor],
    Optional[Tensor],
]

TRAINING = "train"
VALIDATION = "val"
TEST = "test"


@PyTorchIEModel.register()
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
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.t_total = t_total
        self.learning_rate = learning_rate
        self.task_learning_rate = task_learning_rate
        self.warmup_proportion = warmup_proportion

        config = AutoConfig.from_pretrained(model_name_or_path)
        if self.is_from_pretrained:
            self.model = AutoModel.from_config(config=config)
        else:
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

        self.f1 = nn.ModuleDict(
            {
                f"stage_{stage}": torchmetrics.F1Score(
                    num_classes=num_classes, ignore_index=ignore_index
                )
                for stage in [TRAINING, VALIDATION, TEST]
            }
        )

    def forward(self, input_: TransformerTextClassificationModelBatchEncoding) -> TransformerTextClassificationModelBatchOutput:  # type: ignore
        output = self.model(**input_)

        hidden_state = output.last_hidden_state

        cls_embeddings = hidden_state[:, 0, :]
        logits = self.classifier(cls_embeddings)

        return {"logits": logits}

    def step(self, stage: str, batch: TransformerTextClassificationModelStepBatchEncoding):  # type: ignore
        input_, target = batch
        assert target is not None, "target has to be available for training"

        logits = self(input_)["logits"]

        loss = self.loss_fct(logits, target)

        self.log(f"{stage}/loss", loss, on_step=(stage == TRAINING), on_epoch=True, prog_bar=True)

        f1 = self.f1[f"stage_{stage}"]
        f1(logits, target)
        self.log(f"{stage}/f1", f1, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def training_step(self, batch: TransformerTextClassificationModelStepBatchEncoding, batch_idx: int):  # type: ignore
        return self.step(stage=TRAINING, batch=batch)

    def validation_step(self, batch: TransformerTextClassificationModelStepBatchEncoding, batch_idx: int):  # type: ignore
        return self.step(stage=VALIDATION, batch=batch)

    def test_step(self, batch: TransformerTextClassificationModelStepBatchEncoding, batch_idx: int):  # type: ignore
        return self.step(stage=TEST, batch=batch)

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
