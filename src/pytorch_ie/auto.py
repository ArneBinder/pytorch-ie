import os
from typing import Any, Dict, Optional

import torch
from huggingface_hub.constants import PYTORCH_WEIGHTS_NAME
from huggingface_hub.file_download import hf_hub_download

from pytorch_ie.core import PyTorchIEModel, TaskModule
from pytorch_ie.core.hf_hub_mixin import PyTorchIEModelHubMixin, PyTorchIETaskmoduleModelHubMixin
from pytorch_ie.pipeline import Pipeline


class AutoTaskModule(PyTorchIETaskmoduleModelHubMixin):
    @classmethod
    def _from_pretrained(
        cls,
        model_id,
        revision,
        cache_dir,
        force_download,
        proxies,
        resume_download,
        local_files_only,
        use_auth_token,
        **module_kwargs,
    ) -> TaskModule:
        class_name = module_kwargs.pop("taskmodule_type")
        clazz = TaskModule.by_name(class_name)  # type: ignore
        return clazz(**module_kwargs)


class AutoModel(PyTorchIEModelHubMixin):
    @classmethod
    def _from_pretrained(
        cls,
        model_id,
        revision,
        cache_dir,
        force_download,
        proxies,
        resume_download,
        local_files_only,
        use_auth_token,
        map_location="cpu",
        strict=False,
        **model_kwargs,
    ) -> PyTorchIEModel:
        """
        Overwrite this method in case you wish to initialize your model in a different way.
        """
        map_location = torch.device(map_location)

        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, PYTORCH_WEIGHTS_NAME)
        else:
            model_file = hf_hub_download(
                repo_id=model_id,
                filename=PYTORCH_WEIGHTS_NAME,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                use_auth_token=use_auth_token,
                local_files_only=local_files_only,
            )

        class_name = model_kwargs.pop("model_type")
        clazz = PyTorchIEModel.by_name(class_name)
        model = clazz(**model_kwargs)

        state_dict = torch.load(model_file, map_location=map_location)
        model.load_state_dict(state_dict, strict=strict)
        model.eval()

        return model


class AutoPipeline:
    @staticmethod
    def from_pretrained(
        pretrained_model_name_or_path: str,
        force_download: bool = False,
        resume_download: bool = False,
        proxies: Dict = None,
        use_auth_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        taskmodule_kwargs: Dict[str, Any] = None,
        model_kwargs: Dict[str, Any] = None,
        device: int = -1,
        binary_output: bool = False,
        **kwargs,
    ) -> Pipeline:
        taskmodule_kwargs = taskmodule_kwargs or {}
        model_kwargs = model_kwargs or {}

        taskmodule = AutoTaskModule.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            use_auth_token=use_auth_token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            **taskmodule_kwargs,
        )

        model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            use_auth_token=use_auth_token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            **model_kwargs,
        )

        return Pipeline(
            taskmodule=taskmodule,
            model=model,
            device=device,
            binary_output=binary_output,
            **kwargs,
        )
