from typing import Any, Tuple

import torch
from transformers import AutoConfig, AutoModelForSeq2SeqLM, BatchEncoding
from transformers.modeling_outputs import Seq2SeqLMOutput
from typing_extensions import TypeAlias

from pytorch_ie.core import PyTorchIEModel
from pytorch_ie.models.interface import RequiresModelNameOrPath

ModelInputType: TypeAlias = BatchEncoding
ModelOutputType: TypeAlias = Seq2SeqLMOutput

ModelStepInputType: TypeAlias = Tuple[ModelInputType]


@PyTorchIEModel.register()
class TransformerSeq2SeqModel(PyTorchIEModel, RequiresModelNameOrPath):
    def __init__(self, model_name_or_path: str, learning_rate: float = 1e-5, **kwargs) -> None:
        super().__init__(**kwargs)

        self.save_hyperparameters()

        self.learning_rate = learning_rate

        if self.is_from_pretrained:
            config = AutoConfig.from_pretrained(model_name_or_path)
            self.model = AutoModelForSeq2SeqLM.from_config(config=config)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

    def forward(self, inputs: ModelInputType) -> ModelOutputType:
        return self.model(**inputs)

    def predict(
        self,
        inputs: Any,
        **kwargs,
    ) -> Any:
        if "labels" in inputs:
            inputs = {k: v for k, v in inputs.items() if k != "labels"}

        return self.model.generate(**inputs, **kwargs)

    def step(self, batch: ModelStepInputType):
        inputs = batch[0]
        output = self.forward(inputs)

        loss = output.loss

        return loss

    def training_step(self, batch: ModelStepInputType, batch_idx):
        loss = self.step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        return loss

    def validation_step(self, batch: ModelStepInputType, batch_idx):
        loss = self.step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch: ModelStepInputType, batch_idx):
        loss = self.step(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
