import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import requests  # type: ignore
import torch
from huggingface_hub.constants import CONFIG_NAME, PYTORCH_WEIGHTS_NAME
from huggingface_hub.file_download import hf_hub_download
from huggingface_hub.hf_api import HfApi
from huggingface_hub.utils import SoftTemporaryDirectory, validate_hf_hub_args

logger = logging.getLogger(__name__)

MODEL_CONFIG_NAME = CONFIG_NAME
TASKMODULE_CONFIG_NAME = "taskmodule_config.json"
MODEL_CONFIG_TYPE_KEY = "model_type"
TASKMODULE_CONFIG_TYPE_KEY = "taskmodule_type"

# Generic variable that is either PieBaseHFHubMixin or a subclass thereof
T = TypeVar("T", bound="PieBaseHFHubMixin")
# recursive type: either bool or dict from str to this type again
TOverride = Union[bool, Dict[str, "TOverride"]]


def dict_update_nested(d: dict, u: dict, override: Optional[TOverride] = None) -> None:
    """
    Update a dictionary with another dictionary, recursively.

    Args:
        d (`dict`):
            The original dictionary to update.
        u (`dict`):
            The dictionary to use for the update.
        override (`bool` or `dict`, *optional*):
            If `True`, override the original dictionary with the new one.
            If `False`, do not override any keys, just return the original dictionary.
            If `None`, merge the dictionaries recursively.
            If a dictionary, recursively merge the dictionaries, using the provided dictionary as the override.
    Returns:
        None
    """
    if isinstance(override, bool):
        if override:
            d.clear()
            d.update(u)
        return
    if override is None:
        override = {}

    for k, v in u.items():
        if isinstance(v, dict) and k in d:
            if not isinstance(d[k], dict):
                raise ValueError(f"Cannot merge {d[k]} and {v} because {d[k]} is not a dict.")
            dict_update_nested(d[k], v, override=override.get(k))
        else:
            d[k] = v


class PieBaseHFHubMixin:
    """
    A generic mixin to integrate ANY machine learning framework with the Hub.

    To integrate your framework, your model class must inherit from this class. Custom logic for saving/loading models
    have to be overwritten in  [`_from_pretrained`] and [`_save_pretrained`]. [`PyTorchModelHubMixin`] is a good example
    of mixin integration with the Hub. Check out our [integration guide](../guides/integrations) for more instructions.
    """

    config_name = MODEL_CONFIG_NAME
    config_type_key = MODEL_CONFIG_TYPE_KEY

    def __init__(self, *args, is_from_pretrained: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_from_pretrained = is_from_pretrained

    @property
    def is_from_pretrained(self):
        return self._is_from_pretrained

    def _config(self) -> Optional[Dict[str, Any]]:
        return None

    @property
    def has_config(self) -> bool:
        return self._config() is not None

    @property
    def config(self) -> Dict[str, Any]:
        return dict(self._config() or {})  # soft-copy to avoid mutating input

    def save_pretrained(
        self,
        save_directory: Union[str, Path],
        *,
        repo_id: Optional[str] = None,
        push_to_hub: bool = False,
        **kwargs,
    ) -> Optional[str]:
        """
        Save weights in local directory.

        Args:
            save_directory (`str` or `Path`):
                Path to directory in which the model weights and configuration will be saved.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Huggingface Hub after saving it.
            repo_id (`str`, *optional*):
                ID of your repository on the Hub. Used only if `push_to_hub=True`. Will default to the folder name if
                not provided.
            kwargs:
                Additional key word arguments passed along to the [`~ModelHubMixin._from_pretrained`] method.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # saving model weights/files
        self._save_pretrained(save_directory)

        # saving config
        if self.has_config:
            (save_directory / self.config_name).write_text(json.dumps(self.config, indent=2))

        if push_to_hub:
            kwargs = kwargs.copy()  # soft-copy to avoid mutating input
            if self.has_config:  # kwarg for `push_to_hub`
                kwargs["config"] = self.config
            if repo_id is None:
                repo_id = save_directory.name  # Defaults to `save_directory` name
            return self.push_to_hub(repo_id=repo_id, **kwargs)
        return None

    def _save_pretrained(self, save_directory: Path) -> None:
        """
        Overwrite this method in subclass to define how to save your model.
        Check out our [integration guide](../guides/integrations) for instructions.

        Args:
            save_directory (`str` or `Path`):
                Path to directory in which the model weights and configuration will be saved.
        """
        raise NotImplementedError

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(
        cls: Type[T],
        pretrained_model_name_or_path: Union[str, Path],
        *,
        force_download: bool = False,
        resume_download: bool = False,
        proxies: Optional[Dict] = None,
        token: Optional[Union[str, bool]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        local_files_only: bool = False,
        revision: Optional[str] = None,
        **model_kwargs,
    ) -> T:
        """
        Download a model from the Huggingface Hub and instantiate it.

        Args:
            pretrained_model_name_or_path (`str`, `Path`):
                - Either the `model_id` (string) of a model hosted on the Hub, e.g. `bigscience/bloom`.
                - Or a path to a `directory` containing model weights saved using
                    [`~transformers.PreTrainedModel.save_pretrained`], e.g., `../path/to/my_model_directory/`.
            revision (`str`, *optional*):
                Revision of the model on the Hub. Can be a branch name, a git tag or any commit id.
                Defaults to the latest commit on `main` branch.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether to force (re-)downloading the model weights and configuration files from the Hub, overriding
                the existing cache.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether to delete incompletely received files. Will attempt to resume the download if such a file exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on every request.
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. By default, it will use the token
                cached when running `huggingface-cli login`.
            cache_dir (`str`, `Path`, *optional*):
                Path to the folder where cached files are stored.
            local_files_only (`bool`, *optional*, defaults to `False`):
                If `True`, avoid downloading the file and return the path to the local cached file if it exists.
            model_kwargs (`Dict`, *optional*):
                Additional kwargs to pass to the model during initialization.
        """
        model_id = pretrained_model_name_or_path
        config_file: Optional[str] = None
        if os.path.isdir(model_id):
            if cls.config_name in os.listdir(model_id):
                config_file = os.path.join(model_id, cls.config_name)
            else:
                logger.warning(f"{cls.config_name} not found in {Path(model_id).resolve()}")
        elif isinstance(model_id, str):
            try:
                config_file = hf_hub_download(
                    repo_id=str(model_id),
                    filename=cls.config_name,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except requests.exceptions.RequestException:
                logger.warning(f"{cls.config_name} not found in HuggingFace Hub.")

        if config_file is not None:
            with open(config_file, encoding="utf-8") as f:
                config = json.load(f)
            model_kwargs.update({"config": config})

        # The value of is_from_pretrained is set to True when the model is loaded from pretrained.
        # Note that the value may be already available in model_kwargs.
        model_kwargs["is_from_pretrained"] = True

        return cls._from_pretrained(
            model_id=str(model_id),
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            token=token,
            **model_kwargs,
        )

    @classmethod
    def _from_pretrained(
        cls: Type[T],
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: bool,
        local_files_only: bool,
        token: Optional[Union[str, bool]],
        **model_kwargs,
    ) -> T:
        """Overwrite this method in subclass to define how to load your model from pretrained.

        Use [`hf_hub_download`] or [`snapshot_download`] to download files from the Hub before loading them. Most
        args taken as input can be directly passed to those 2 methods. If needed, you can add more arguments to this
        method using "model_kwargs". For example [`PyTorchModelHubMixin._from_pretrained`] takes as input a `map_location`
        parameter to set on which device the model should be loaded.

        Check out our [integration guide](../guides/integrations) for more instructions.

        Args:
            model_id (`str`):
                ID of the model to load from the Huggingface Hub (e.g. `bigscience/bloom`).
            revision (`str`, *optional*):
                Revision of the model on the Hub. Can be a branch name, a git tag or any commit id. Defaults to the
                latest commit on `main` branch.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether to force (re-)downloading the model weights and configuration files from the Hub, overriding
                the existing cache.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether to delete incompletely received files. Will attempt to resume the download if such a file exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint (e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`).
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. By default, it will use the token
                cached when running `huggingface-cli login`.
            cache_dir (`str`, `Path`, *optional*):
                Path to the folder where cached files are stored.
            local_files_only (`bool`, *optional*, defaults to `False`):
                If `True`, avoid downloading the file and return the path to the local cached file if it exists.
            model_kwargs:
                Additional keyword arguments passed along to the [`~ModelHubMixin._from_pretrained`] method.
        """
        raise NotImplementedError

    @validate_hf_hub_args
    def push_to_hub(
        self,
        repo_id: str,
        *,
        config: Optional[dict] = None,
        commit_message: str = "Push model using huggingface_hub.",
        private: bool = False,
        api_endpoint: Optional[str] = None,
        token: Optional[str] = None,
        branch: Optional[str] = None,
        create_pr: Optional[bool] = None,
        allow_patterns: Optional[Union[List[str], str]] = None,
        ignore_patterns: Optional[Union[List[str], str]] = None,
        delete_patterns: Optional[Union[List[str], str]] = None,
    ) -> str:
        """
        Upload model checkpoint to the Hub.

        Use `allow_patterns` and `ignore_patterns` to precisely filter which files should be pushed to the hub. Use
        `delete_patterns` to delete existing remote files in the same commit. See [`upload_folder`] reference for more
        details.


        Args:
            repo_id (`str`):
                ID of the repository to push to (example: `"username/my-model"`).
            config (`dict`, *optional*):
                Configuration object to be saved alongside the model weights.
            commit_message (`str`, *optional*):
                Message to commit while pushing.
            private (`bool`, *optional*, defaults to `False`):
                Whether the repository created should be private.
            api_endpoint (`str`, *optional*):
                The API endpoint to use when pushing the model to the hub.
            token (`str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. By default, it will use the token
                cached when running `huggingface-cli login`.
            branch (`str`, *optional*):
                The git branch on which to push the model. This defaults to `"main"`.
            create_pr (`boolean`, *optional*):
                Whether or not to create a Pull Request from `branch` with that commit. Defaults to `False`.
            allow_patterns (`List[str]` or `str`, *optional*):
                If provided, only files matching at least one pattern are pushed.
            ignore_patterns (`List[str]` or `str`, *optional*):
                If provided, files matching any of the patterns are not pushed.
            delete_patterns (`List[str]` or `str`, *optional*):
                If provided, remote files matching any of the patterns will be deleted from the repo.

        Returns:
            The url of the commit of your model in the given repository.
        """
        api = HfApi(endpoint=api_endpoint, token=token)
        repo_id = api.create_repo(repo_id=repo_id, private=private, exist_ok=True).repo_id

        # Push the files to the repo in a single commit
        with SoftTemporaryDirectory() as tmp:
            saved_path = Path(tmp) / repo_id
            self.save_pretrained(saved_path, config=config)
            return api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=saved_path,
                commit_message=commit_message,
                revision=branch,
                create_pr=create_pr,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                delete_patterns=delete_patterns,
            )

    @classmethod
    def from_config(cls, config: dict, **kwargs) -> "PieBaseHFHubMixin":
        """
        Instantiate from a configuration object.

        Args:
            config (`dict`):
                The configuration object to instantiate.
            kwargs:
                Additional keyword arguments passed along to the specific model class.
        """
        config = config.copy()
        # remove config_type_key entry, e.g. model_type, from config
        config.pop(cls.config_type_key, None)
        return cls._from_config(config=config, **kwargs)

    @classmethod
    def _from_config(
        cls: Type[T], config: dict, config_override: Optional[TOverride] = None, **kwargs
    ) -> T:
        """
        Instantiate from a configuration object.

        Args:
            config (`dict`):
                The configuration object to instantiate.
            kwargs:
                Additional keyword arguments passed along to the specific model class.
        """
        config = config.copy()
        dict_update_nested(config, kwargs, override=config_override)
        return cls(**config)


TModel = TypeVar("TModel", bound="PieModelHFHubMixin")


class PieModelHFHubMixin(PieBaseHFHubMixin):
    config_name = MODEL_CONFIG_NAME
    config_type_key = MODEL_CONFIG_TYPE_KEY
    weights_file_name = PYTORCH_WEIGHTS_NAME

    """
    Implementation of [`ModelHubMixin`] to provide model Hub upload/download capabilities to PyTorch models. The model
    is set in evaluation mode by default using `model.eval()` (dropout modules are deactivated). To train the model,
    you should first set it back in training mode with `model.train()`.

    Example:

    ```python
    >>> import torch
    >>> import torch.nn as nn
    >>> from huggingface_hub import PyTorchModelHubMixin


    >>> class MyModel(nn.Module, PyTorchModelHubMixin):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.param = nn.Parameter(torch.rand(3, 4))
    ...         self.linear = nn.Linear(4, 5)

    ...     def forward(self, x):
    ...         return self.linear(x + self.param)
    >>> model = MyModel()

    # Save model weights to local directory
    >>> model.save_pretrained("my-awesome-model")

    # Push model weights to the Hub
    >>> model.push_to_hub("my-awesome-model")

    # Download and initialize weights from the Hub
    >>> model = MyModel.from_pretrained("username/my-awesome-model")
    ```
    """

    def save_model_file(self, model_file: str) -> None:
        """Save weights from a Pytorch model to a local directory."""
        model_to_save = self.module if hasattr(self, "module") else self  # type: ignore
        torch.save(model_to_save.state_dict(), model_file)

    def load_model_file(
        self, model_file: str, map_location: str = "cpu", strict: bool = False
    ) -> None:
        state_dict = torch.load(model_file, map_location=torch.device(map_location))
        self.load_state_dict(state_dict, strict=strict)  # type: ignore
        self.eval()  # type: ignore

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights from a Pytorch model to a local directory."""
        self.save_model_file(str(save_directory / self.weights_file_name))

    @classmethod
    def _from_pretrained(
        cls: Type[TModel],
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
        config_override: Optional[TOverride] = None,
        **model_kwargs,
    ) -> TModel:

        config = (config or {}).copy()
        dict_update_nested(config, model_kwargs, override=config_override)
        if cls.config_type_key is not None:
            config.pop(cls.config_type_key)
        model = cls(**config)

        """Load Pytorch pretrained weights and return the loaded model."""
        if os.path.isdir(model_id):
            logger.info("Loading weights from local directory")
            model_file = os.path.join(model_id, model.weights_file_name)
        else:
            model_file = hf_hub_download(
                repo_id=model_id,
                filename=model.weights_file_name,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )

        model.load_model_file(model_file, map_location=map_location, strict=strict)

        return model


TTaskModule = TypeVar("TTaskModule", bound="PieTaskModuleHFHubMixin")


class PieTaskModuleHFHubMixin(PieBaseHFHubMixin):
    config_name = TASKMODULE_CONFIG_NAME
    config_type_key = TASKMODULE_CONFIG_TYPE_KEY

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _save_pretrained(self, save_directory):
        return None

    @classmethod
    def _from_pretrained(
        cls: Type[TTaskModule],
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
        config_override: Optional[TOverride] = None,
        **taskmodule_kwargs,
    ) -> TTaskModule:
        config = (config or {}).copy()
        dict_update_nested(config, taskmodule_kwargs, override=config_override)
        if cls.config_type_key is not None:
            config.pop(cls.config_type_key)

        taskmodule = cls(**config)
        return taskmodule
