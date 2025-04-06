from typing import Any, Dict

import torch
from pie_core import Registrable
from pie_core.hf_hub_mixin import PieModelHFHubMixin
from pytorch_lightning import LightningModule


class PyTorchIEModel(PieModelHFHubMixin, LightningModule, Registrable):

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

    def _config(self) -> Dict[str, Any]:
        config = super()._config() or {}
        config[self.config_type_key] = PyTorchIEModel.name_for_object_class(self)
        # add all hparams
        config.update(self.hparams)
        return config

    def decode(self, inputs: Any, outputs: Any) -> Any:
        return outputs

    def predict(self, inputs: Any, **kwargs) -> Any:
        outputs = self(inputs, **kwargs)
        decoded_outputs = self.decode(inputs=inputs, outputs=outputs)
        return decoded_outputs


class AutoPyTorchIEModel(PieModelHFHubMixin):

    @classmethod
    def from_config(cls, config: dict, **kwargs) -> PyTorchIEModel:
        """Build a model from a config dict."""
        config = config.copy()
        class_name = config.pop(cls.config_type_key)
        # the class name may be overridden by the kwargs
        class_name = kwargs.pop(cls.config_type_key, class_name)
        clazz = PyTorchIEModel.by_name(class_name)
        return clazz._from_config(config, **kwargs)
