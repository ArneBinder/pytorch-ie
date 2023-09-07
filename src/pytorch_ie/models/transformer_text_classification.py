import logging
from typing import Any, Dict, MutableMapping, Optional, Tuple

import torchmetrics
from torch import Tensor, nn
from torch.optim import AdamW
from transformers import AutoConfig, AutoModel, get_linear_schedule_with_warmup
from typing_extensions import TypeAlias

from pytorch_ie.core import PyTorchIEModel
from pytorch_ie.models.interface import RequiresModelNameOrPath, RequiresNumClasses

ModelInputType: TypeAlias = MutableMapping[str, Any]
ModelOutputType: TypeAlias = Dict[str, Any]

ModelStepInputType = Tuple[
    ModelInputType,
    Optional[Tensor],
]

TRAINING = "train"
VALIDATION = "val"
TEST = "test"

logger = logging.getLogger(__name__)


@PyTorchIEModel.register()
class TransformerTextClassificationModel(
    PyTorchIEModel, RequiresModelNameOrPath, RequiresNumClasses
):
    def __init__(
        self,
        model_name_or_path: str,
        num_classes: int,
        tokenizer_vocab_size: Optional[int] = None,
        ignore_index: Optional[int] = None,
        learning_rate: float = 1e-5,
        task_learning_rate: float = 1e-4,
        warmup_proportion: float = 0.1,
        freeze_model: bool = False,
        multi_label: bool = False,
        t_total: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if t_total is not None:
            logger.warning(
                "t_total is deprecated, we use estimated_stepping_batches from the pytorch lightning trainer instead"
            )

        self.save_hyperparameters(ignore=["t_total"])

        self.learning_rate = learning_rate
        self.task_learning_rate = task_learning_rate
        self.warmup_proportion = warmup_proportion

        config = AutoConfig.from_pretrained(model_name_or_path)
        if self.is_from_pretrained:
            self.model = AutoModel.from_config(config=config)
        else:
            self.model = AutoModel.from_pretrained(model_name_or_path, config=config)

        if freeze_model:
            for param in self.model.parameters():
                param.requires_grad = False

        if tokenizer_vocab_size is not None:
            self.model.resize_token_embeddings(tokenizer_vocab_size)

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
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    task="multilabel" if multi_label else "multiclass",
                )
                for stage in [TRAINING, VALIDATION, TEST]
            }
        )

    def forward(self, inputs: ModelInputType) -> ModelOutputType:
        output = self.model(**inputs)

        hidden_state = output.last_hidden_state

        cls_embeddings = hidden_state[:, 0, :]
        logits = self.classifier(cls_embeddings)

        return {"logits": logits}

    def step(self, stage: str, batch: ModelStepInputType):
        inputs, target = batch
        assert target is not None, "target has to be available for training"

        logits = self(inputs)["logits"]

        loss = self.loss_fct(logits, target)

        self.log(f"{stage}/loss", loss, on_step=(stage == TRAINING), on_epoch=True, prog_bar=True)

        f1 = self.f1[f"stage_{stage}"]
        f1(logits, target)
        self.log(f"{stage}/f1", f1, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def training_step(self, batch: ModelStepInputType, batch_idx: int):
        return self.step(stage=TRAINING, batch=batch)

    def validation_step(self, batch: ModelStepInputType, batch_idx: int):
        return self.step(stage=VALIDATION, batch=batch)

    def test_step(self, batch: ModelStepInputType, batch_idx: int):
        return self.step(stage=TEST, batch=batch)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        if self.warmup_proportion > 0.0:
            stepping_batches = self.trainer.estimated_stepping_batches
            scheduler = get_linear_schedule_with_warmup(
                optimizer, int(stepping_batches * self.warmup_proportion), stepping_batches
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        else:
            return optimizer
