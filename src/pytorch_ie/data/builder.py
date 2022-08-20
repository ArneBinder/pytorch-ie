import abc
from typing import Any, Dict, Mapping, Optional, Type

from datasets.load import load_dataset_builder

import datasets
from pytorch_ie.core.document import Document
from pytorch_ie.data.dataset import Dataset, decorate_convert_to_dict_of_lists


class GeneratorBasedBuilder(datasets.builder.GeneratorBasedBuilder):
    DOCUMENT_TYPE: Optional[Type[Document]] = None

    BASE_DATASET_PATH: Optional[str] = None

    # Define a mapping from config names to base config names. If the selected config name
    # is not in the mapping, that name will be reused as base config name. The whole mapping
    # can also be set to None to disable any passing of config names to the base dataset builder.
    CONFIG_NAME_MAPPING: Optional[Dict[str, str]] = {}

    # Define kwargs to create base configs from scratch. This should contain config names as keys
    # (they will come from the result of CONFIG_NAME_MAPPING or from base_dataset_kwargs["config_name"])
    # and the respective config kwargs dicts as values. Per default, the config name will be removed
    # from the base builder kwargs, if it is used here because Huggingface datasets allows to either
    # get configs by name or newly construct them. However, just add the name here if you want to keep
    # it, i.e. BASE_DATASET_KWARGS_DICT["your_base_name"]["name"] = "your_base_name". This is useful
    # if you just want to add some builder kwargs (e.g. setting data_dir or revision), but without the
    # intention to create the base config from scratch.
    BASE_DATASET_KWARGS_DICT: Optional[Dict[Optional[str], Dict[str, Any]]] = None

    def __init__(self, base_dataset_kwargs: Optional[Dict[str, Any]] = None, **kwargs):

        base_dataset_kwargs = base_dataset_kwargs or {}
        self.base_builder = None
        if self.BASE_DATASET_PATH is not None:
            base_builder_kwargs: Dict[str, Any] = {}
            if self.CONFIG_NAME_MAPPING is not None:
                config_name = kwargs.get("config_name", None)
                base_builder_kwargs["name"] = self.CONFIG_NAME_MAPPING.get(
                    config_name, config_name
                )
            # move the config name (this may be None) over from base_dataset_kwargs already here to allow access via
            # logic for BASE_DATASET_KWARGS_DICT
            if "name" in base_dataset_kwargs:
                base_builder_kwargs["name"] = base_dataset_kwargs.pop("name")
            # use values from BASE_DATASET_KWARGS_DICT as defaults, but allow to overwrite them later on via
            # base_dataset_kwargs (except for config name).
            if self.BASE_DATASET_KWARGS_DICT is not None:
                # get the default base builder kwargs (can contain base config kwargs) from BASE_DATASET_KWARGS_DICT
                default_base_builder_kwargs = {}
                # Note that we remove the config name from the kwargs since either a name or config_kwargs,
                # but not both, should be present when creating the dataset builder. However, you can add
                # the name manually by setting the respective entry, i.e.
                # BASE_DATASET_KWARGS_DICT["selected_base_config"]["name"] = "selected_base_config".
                config_name = base_builder_kwargs.pop("name", None)
                if config_name in self.BASE_DATASET_KWARGS_DICT:
                    default_base_builder_kwargs = self.BASE_DATASET_KWARGS_DICT[config_name]
                base_builder_kwargs.update(default_base_builder_kwargs)
            base_builder_kwargs.update(base_dataset_kwargs)
            self.base_builder = load_dataset_builder(
                path=self.BASE_DATASET_PATH,
                **base_builder_kwargs,
            )

        super().__init__(**kwargs)

    def _info(self):
        return self.base_builder._info()

    def _split_generators(self, dl_manager):
        return self.base_builder._split_generators(dl_manager)

    def _generate_examples(self, *args, **kwargs):
        return self.base_builder._generate_examples(*args, **kwargs)

    @abc.abstractmethod
    def _generate_document(self, example, dataset):
        pass

    def _generate_document_kwargs(self, dataset):
        return None

    def _post_process(
        self, dataset: datasets.Dataset, resources_paths: Mapping[str, str]
    ) -> Optional[datasets.Dataset]:
        fn_kwargs = {}
        additional_kwargs = self._generate_document_kwargs(dataset)

        if additional_kwargs is not None:
            fn_kwargs.update(additional_kwargs)

        mapped_dataset = dataset.map(
            decorate_convert_to_dict_of_lists(self._generate_document), fn_kwargs=fn_kwargs
        )

        document_dataset = Dataset.from_hf_dataset(
            mapped_dataset, document_type=self.DOCUMENT_TYPE
        )

        return document_dataset
