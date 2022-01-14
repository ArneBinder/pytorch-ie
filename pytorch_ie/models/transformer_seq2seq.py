from typing import Any, Dict

import torch
from transformers import AutoModelForSeq2SeqLM, BatchEncoding

from pytorch_ie.core.pytorch_ie import PyTorchIEModel

TransformerSeq2SeqModelBatchEncoding = BatchEncoding
TransformerSeq2SeqModelBatchOutput = Dict[str, Any]


class TransformerSeq2SeqModel(PyTorchIEModel):
    def __init__(
        self,
        model_name_or_path: str,
        learning_rate: float = 1e-5,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.learning_rate = learning_rate

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

    def forward(
        self,
        inputs: TransformerSeq2SeqModelBatchEncoding,
    ) -> TransformerSeq2SeqModelBatchOutput:
        return self.model(**inputs)

    def predict(
        self,
        inputs: Any,
        **kwargs,
    ) -> Any:
        # TODO: check if this is necessary
        if "labels" in inputs:
            inputs = {k: v for k, v in inputs.items() if k != "labels"}

        return self.model.generate(**inputs, **kwargs)

    def step(self, batch: Any):
        inputs = batch
        output = self.forward(inputs)

        loss = output.loss

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)