import os
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

import torch
from huggingface_hub.constants import PYTORCH_WEIGHTS_NAME
from huggingface_hub.file_download import hf_hub_download

from pytorch_ie.core import PyTorchIEModel, TaskModule
from pytorch_ie.core.hf_hub_mixin import PieModelHFHubMixin, PieTaskModuleHFHubMixin
from pytorch_ie.pipeline import Pipeline


class AutoModel(PieModelHFHubMixin):
    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: bool,
        local_files_only: bool,
        token: Union[str, bool, None],
        map_location: str = "cpu",
        strict: bool = False,
        config: Optional[dict] = None,
        **model_kwargs,
    ) -> PyTorchIEModel:
        """
        Overwrite this method in case you wish to initialize your model in a different way.
        """

        """Load Pytorch pretrained weights and return the loaded model."""
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
                token=token,
                local_files_only=local_files_only,
            )

        config = (config or {}).copy()
        config.update(model_kwargs)
        class_name = config.pop(cls.config_type_key)
        clazz = PyTorchIEModel.by_name(class_name)
        model = clazz(**config)

        state_dict = torch.load(model_file, map_location=torch.device(map_location))
        model.load_state_dict(state_dict, strict=strict)  # type: ignore
        model.eval()  # type: ignore

        return model


class AutoTaskModule(PieTaskModuleHFHubMixin):
    @classmethod
    def _from_pretrained(  # type: ignore
        cls,
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: bool,
        local_files_only: bool,
        token: Union[str, bool, None],
        map_location: str = "cpu",
        strict: bool = False,
        config: Optional[dict] = None,
        **taskmodule_kwargs,
    ) -> TaskModule:
        config = (config or {}).copy()
        config.update(taskmodule_kwargs)
        class_name = config.pop(cls.config_type_key)
        clazz: Type[TaskModule] = TaskModule.by_name(class_name)
        taskmodule = clazz(**config)
        taskmodule.post_prepare()
        return taskmodule


class AutoPipeline:
    @staticmethod
    def from_pretrained(
        pretrained_model_name_or_path: str,
        force_download: bool = False,
        resume_download: bool = False,
        proxies: Optional[Dict] = None,
        use_auth_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        taskmodule_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
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
