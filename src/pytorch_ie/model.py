from typing import Any, Dict

import torch
from pie_core import Auto, Model
from pytorch_lightning import LightningModule


class PyTorchIEModel(Model, LightningModule):
    weights_file_name = "pytorch_model.bin"

    def _config(self) -> Dict[str, Any]:
        config = super()._config() or {}
        # add all hparams
        config.update(self.hparams)
        return config

    def save_model_file(self, model_file: str) -> None:
        """Save weights from a Pytorch model to a local directory."""
        model_to_save: torch.nn.Module = self.module if hasattr(self, "module") else self
        torch.save(model_to_save.state_dict(), model_file)

    def load_model_file(
        self, model_file: str, map_location: str = "cpu", strict: bool = False
    ) -> None:
        state_dict = torch.load(model_file, map_location=torch.device(map_location))
        self.load_state_dict(state_dict, strict=strict)
        # The model is set in evaluation mode by default using `model.eval()`
        # (dropout modules are deactivated). To train the model, you should first
        # set it back in training mode with `model.train()`. This is especially
        # important when using pytorch-lightning >= 2.2.0, as it maintains the
        # training/evaluation state of the model when training via `fit()`.
        self.eval()

    def decode(self, inputs: Any, outputs: Any) -> Any:
        return outputs

    def predict(self, inputs: Any, **kwargs) -> Any:
        outputs = self(inputs, **kwargs)
        decoded_outputs = self.decode(inputs=inputs, outputs=outputs)
        return decoded_outputs


# TODO: remove this class when all models are registered with @Model.register()
#   also see notes in PyTorchIEPipeline
class AutoPyTorchIEModel(Model, Auto[PyTorchIEModel]):

    BASE_CLASS = PyTorchIEModel
