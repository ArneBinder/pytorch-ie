import abc
from functools import partial
from typing import Any, Dict, Mapping, Optional, Type

from datasets.load import load_dataset_builder

import datasets
from pytorch_ie.core.document import Document
from pytorch_ie.data.dataset import Dataset, IterableDataset, decorate_convert_to_dict_of_lists


class GeneratorBasedBuilder(datasets.builder.GeneratorBasedBuilder):
    DOCUMENT_TYPE: Optional[Type[Document]] = None

    BASE_DATASET_PATH: Optional[str] = None

    # Define kwargs to create base configs. This should contain config names as keys
    # and the respective config kwargs dicts as values. If the config name is not contained, a new entry
    # {"name": config_name} will be created for it, i.e. the config name is passed as base config name.
    # This default behaviour can be disabled by setting BASE_CONFIG_KWARGS_DICT to None.
    BASE_CONFIG_KWARGS_DICT: Optional[Dict[Optional[str], Dict[str, Any]]] = {}
    # Define base builder kwargs. This should contain config names as keys and the respective
    # builder kwargs dicts as values.
    BASE_BUILDER_KWARGS_DICT: Optional[Dict[Optional[str], Dict[str, Any]]] = None

    def __init__(self, base_dataset_kwargs: Optional[Dict[str, Any]] = None, **kwargs):

        self.base_builder = None
        if self.BASE_DATASET_PATH is not None:
            base_dataset_kwargs = base_dataset_kwargs or {}
            base_builder_kwargs: Dict[str, Any] = {}

            config_name = kwargs.get("config_name", None)

            # get base config kwargs from mapping
            if self.BASE_CONFIG_KWARGS_DICT is not None:
                if config_name in self.BASE_CONFIG_KWARGS_DICT:
                    config_kwargs = self.BASE_CONFIG_KWARGS_DICT[config_name]
                else:
                    # if the config name is not in BASE_CONFIG_KWARGS_DICT,
                    # we pass it as base config name
                    config_kwargs = {"name": config_name}
                base_builder_kwargs.update(config_kwargs)

            # get base builder kwargs from mapping
            if self.BASE_BUILDER_KWARGS_DICT is not None:
                base_builder_kwargs.update(self.BASE_BUILDER_KWARGS_DICT[config_name])

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

        if self.DOCUMENT_TYPE is None:
            raise TypeError("the builder has no DOCUMENT_TYPE defined")

        document_dataset = Dataset.from_hf_dataset(
            mapped_dataset, document_type=self.DOCUMENT_TYPE
        )

        return document_dataset

    def _as_streaming_dataset_single(
        self,
        splits_generator,
    ) -> IterableDataset:
        dataset = super()._as_streaming_dataset_single(splits_generator)

        fn = decorate_convert_to_dict_of_lists(self._generate_document)
        fn_kwargs = self._generate_document_kwargs(dataset)
        if fn_kwargs is not None:
            fn = partial(fn, **fn_kwargs)
        mapped_dataset = dataset.map(fn)

        if self.DOCUMENT_TYPE is None:
            raise TypeError("the builder has no DOCUMENT_TYPE defined")

        return IterableDataset.from_hf_dataset(
            dataset=mapped_dataset, document_type=self.DOCUMENT_TYPE
        )
