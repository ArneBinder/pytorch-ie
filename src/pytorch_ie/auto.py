from typing import Any, Dict, Optional, Type

from pytorch_ie.core import PyTorchIEModel, TaskModule
from pytorch_ie.core.hf_hub_mixin import PieModelHFHubMixin, PieTaskModuleHFHubMixin
from pytorch_ie.pipeline import Pipeline


class AutoModel(PieModelHFHubMixin):

    @classmethod
    def from_config(cls, config: dict, **kwargs) -> PyTorchIEModel:
        """Build a model from a config dict."""
        config = config.copy()
        class_name = config.pop(cls.config_type_key)
        # the class name may be overridden by the kwargs
        class_name = kwargs.pop(cls.config_type_key, class_name)
        clazz = PyTorchIEModel.by_name(class_name)
        return clazz._from_config(config, **kwargs)


class AutoTaskModule(PieTaskModuleHFHubMixin):

    @classmethod
    def from_config(cls, config: dict, **kwargs) -> TaskModule:  # type: ignore
        """Build a task module from a config dict."""
        config = config.copy()
        class_name = config.pop(cls.config_type_key)
        # the class name may be overridden by the kwargs
        class_name = kwargs.pop(cls.config_type_key, class_name)
        clazz: Type[TaskModule] = TaskModule.by_name(class_name)
        return clazz._from_config(config, **kwargs)


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
