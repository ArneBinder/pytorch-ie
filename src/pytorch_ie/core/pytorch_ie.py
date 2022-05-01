from typing import Any, Dict

from pytorch_lightning import LightningModule

from pytorch_ie.core.hf_hub_mixin import PyTorchIEModelHubMixin
from pytorch_ie.core.registerable import Registrable


class PyTorchIEModel(LightningModule, Registrable, PyTorchIEModelHubMixin):
    def _config(self) -> Dict[str, Any]:
        config = dict(self.hparams)
        config["model_type"] = self.__class__.__name__
        return config

    def predict(
        self,
        inputs: Any,
        **kwargs,
    ) -> Any:
        return self(inputs, **kwargs)
