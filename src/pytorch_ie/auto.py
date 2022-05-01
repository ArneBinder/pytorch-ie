from typing import Dict, Optional
from pytorch_ie.taskmodules import TaskModule
from pytorch_ie.core.hf_hub_mixin import PyTorchIETaskmoduleModelHubMixin


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
    ):
        class_name = module_kwargs.pop("taskmodule_type")
        clazz = TaskModule.by_name(class_name)
        return clazz(**module_kwargs)
