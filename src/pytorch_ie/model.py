from typing import Any

import torch
from pie_core import Auto, Model
from pytorch_lightning import LightningModule


class PyTorchIEModel(Model, LightningModule):

    def save_model_file(self, model_file: str) -> None:
        """Save weights from a Pytorch model to a local directory."""
        model_to_save: torch.nn.Module = self.module if hasattr(self, "module") else self
        torch.save(model_to_save.state_dict(), model_file)

    def load_model_file(
        self, model_file: str, map_location: str = "cpu", strict: bool = False
    ) -> None:
        state_dict = torch.load(model_file, map_location=torch.device(map_location))
        self.load_state_dict(state_dict, strict=strict)
        self.eval()

    def decode(self, inputs: Any, outputs: Any) -> Any:
        return outputs

    def predict(self, inputs: Any, **kwargs) -> Any:
        outputs = self(inputs, **kwargs)
        decoded_outputs = self.decode(inputs=inputs, outputs=outputs)
        return decoded_outputs


class AutoPyTorchIEModel(Model, Auto[PyTorchIEModel]):

    BASE_CLASS = PyTorchIEModel
