from typing import Any, Dict, Optional, Tuple

import torch
import torchmetrics
from torch import Tensor, nn
from transformers import AutoConfig, AutoModelForTokenClassification, BatchEncoding
from typing_extensions import TypeAlias

from pytorch_ie.model import PyTorchIEModel
from pytorch_ie.models.interface import RequiresModelNameOrPath, RequiresNumClasses

ModelInputType: TypeAlias = BatchEncoding
ModelOutputType: TypeAlias = Dict[str, Any]

ModelStepInputType: TypeAlias = Tuple[
    ModelInputType,
    Optional[Tensor],
]


TRAINING = "train"
VALIDATION = "val"
TEST = "test"


@PyTorchIEModel.register()
class TransformerTokenClassificationModel(
    PyTorchIEModel, RequiresModelNameOrPath, RequiresNumClasses
):
    def __init__(
        self,
        model_name_or_path: str,
        num_classes: int,
        learning_rate: float = 1e-5,
        label_pad_token_id: int = -100,
        ignore_index: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.label_pad_token_id = label_pad_token_id
        self.num_classes = num_classes

        config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_classes)
        if self.is_from_pretrained:
            self.model = AutoModelForTokenClassification.from_config(config=config)
        else:
            self.model = AutoModelForTokenClassification.from_pretrained(
                model_name_or_path, config=config
            )

        self.f1 = nn.ModuleDict(
            {
                f"stage_{stage}": torchmetrics.F1Score(
                    num_classes=num_classes, ignore_index=ignore_index, task="multiclass"
                )
                for stage in [TRAINING, VALIDATION, TEST]
            }
        )

    def forward(self, inputs: ModelInputType) -> ModelOutputType:
        return self.model(**inputs)

    def step(
        self,
        stage: str,
        batch: ModelStepInputType,
    ):
        inputs, target = batch
        assert target is not None, "target has to be available for training"

        inputs["labels"] = target
        output = self(inputs)

        loss = output.loss
        # show loss on each step only during training
        self.log(f"{stage}/loss", loss, on_step=(stage == TRAINING), on_epoch=True, prog_bar=True)

        target_flat = target.view(-1)

        valid_indices = target_flat != self.label_pad_token_id
        valid_logits = output.logits.view(-1, self.num_classes)[valid_indices]
        valid_target = target_flat[valid_indices]

        f1 = self.f1[f"stage_{stage}"]
        f1(valid_logits, valid_target)
        self.log(f"{stage}/f1", f1, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def training_step(self, batch: ModelStepInputType, batch_idx: int):
        return self.step(stage=TRAINING, batch=batch)

    def validation_step(self, batch: ModelStepInputType, batch_idx: int):
        return self.step(stage=VALIDATION, batch=batch)

    def test_step(self, batch: ModelStepInputType, batch_idx: int):
        return self.step(stage=TEST, batch=batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
