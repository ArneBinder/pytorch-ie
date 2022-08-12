import abc
from typing import Any, Dict, Mapping, Optional, Type

from datasets.load import load_dataset_builder

import datasets
from pytorch_ie.core.document import Document
from pytorch_ie.data.dataset import Dataset, decorate_convert_to_dict_of_lists


class GeneratorBasedBuilder(datasets.builder.GeneratorBasedBuilder):
    DOCUMENT_TYPE: Optional[Type[Document]] = None

    BASE_DATASET_PATH: Optional[str] = None

    # Define further arguments for the base dataset like a config name or revision.
    # See datasets.load.load_dataset_builder for all possible parameters.
    BASE_DATASET_KWARGS: Optional[Dict[str, Any]] = None

    # Define a mapping from config names to base config names. If there is no key defined for the
    # selected config, that will be reused as base config name. Can also be set to None to disable
    # any passing of config names to the base dataset builder.
    CONFIG_NAME_MAPPING: Optional[Dict[str, str]] = {}

    def __init__(self, base_dataset_kwargs: Optional[Dict[str, Any]] = None, **kwargs):

        self.base_builder = None
        if self.BASE_DATASET_PATH is not None:
            base_builder_kwargs = {}
            # use values from BASE_DATASET_KWARGS as defaults, but allow to overwrite them
            if self.BASE_DATASET_KWARGS is not None:
                base_builder_kwargs.update(self.BASE_DATASET_KWARGS)
            if self.CONFIG_NAME_MAPPING is not None:
                config_name = kwargs.get("config_name", None)
                base_builder_kwargs["name"] = self.CONFIG_NAME_MAPPING.get(
                    config_name, config_name
                )
            if base_dataset_kwargs is not None:
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
