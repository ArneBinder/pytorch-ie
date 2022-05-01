from pytorch_ie.taskmodules import TaskModule
from pytorch_ie.core.hf_hub_mixin import PyTorchIETaskmoduleModelHubMixin, PyTorchIEModelHubMixin
import os
import torch
from huggingface_hub.constants import PYTORCH_WEIGHTS_NAME
from huggingface_hub.file_download import hf_hub_download
from pytorch_ie.core import PyTorchIEModel


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
    ):
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
