from typing import Any, Dict

from pytorch_lightning import LightningModule

from pytorch_ie.core.hf_hub_mixin import PyTorchIEModelHubMixin
from pytorch_ie.core.registrable import Registrable


class PyTorchIEModel(PyTorchIEModelHubMixin, LightningModule, Registrable):
    def _config(self) -> Dict[str, Any]:
        config = dict(self.hparams)
        this_class = self.__class__
        registered_name = PyTorchIEModel.registered_name_for_class(this_class)
        config["model_type"] = (
            registered_name if registered_name is not None else this_class.__name__
        )
        return config

    def predict(
        self,
        inputs: Any,
        **kwargs,
    ) -> Any:
        return self(inputs, **kwargs)
