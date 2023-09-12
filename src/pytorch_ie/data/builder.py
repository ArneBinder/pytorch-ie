import abc
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, TypeVar, Union, overload

import datasets as hf_datasets

from pytorch_ie.core.document import Document
from pytorch_ie.data.dataset import (
    Dataset,
    DocumentConvertersType,
    IterableDataset,
    decorate_convert_to_dict_of_lists,
    get_pie_dataset_type,
)
from pytorch_ie.utils.hydra import resolve_target


def get_general_dataset_builder_parent_class(
    obj: hf_datasets.builder.DatasetBuilder,
) -> Type[hf_datasets.builder.DatasetBuilder]:
    general_dataset_builder_parent_classes = [
        cls
        for cls in hf_datasets.builder.DatasetBuilder.__subclasses__()
        if cls != PieDatasetBuilder and isinstance(obj, cls)
    ]
    if len(general_dataset_builder_parent_classes) != 1:
        raise TypeError("can not determine general dataset builder parent class of the object")
    return general_dataset_builder_parent_classes[0]


class PieDatasetBuilder(hf_datasets.builder.DatasetBuilder):
    # The default pytorch-ie document type for the dataset.
    DOCUMENT_TYPE: Optional[Type[Document]] = None
    # A mapping from config names to PIE document types. Use this to specify individual
    # document types per config.
    DOCUMENT_TYPES: Dict[str, Type[Document]] = {}

    # The default path to the Huggingface dataset loading script that will be used as base dataset.
    BASE_DATASET_PATH: Optional[str] = None
    # A mapping from config names to Huggingface dataset loading script paths. Use this to specify individual
    # base datasets for each config.
    BASE_DATASET_PATHS: Dict[str, str] = {}

    # Define kwargs to create base configs. This should contain config names as keys
    # and the respective config kwargs dicts as values. If the config name is not contained, a new entry
    # {"name": config_name} will be created for it, i.e. the config name is passed as base config name.
    # This default behaviour can be disabled by setting BASE_CONFIG_KWARGS_DICT to None.
    BASE_CONFIG_KWARGS_DICT: Optional[Dict[Optional[str], Dict[str, Any]]] = {}
    # Define base builder kwargs. This should contain config names as keys and the respective
    # builder kwargs dicts as values.
    BASE_BUILDER_KWARGS_DICT: Optional[Dict[Optional[str], Dict[str, Any]]] = None

    # Define document converters. This should be a mapping from document types as keys to the respective
    # document converters as values. The document converters can be either callables or dicts
    # that map from original field names to new field names. If a callable is provided, it will be used to
    # convert the document. If a dict is provided, it will be used to rename the fields of the
    # document (this is done by renaming the columns which is much more efficient).
    DOCUMENT_CONVERTERS: DocumentConvertersType = {}

    def __init__(
        self,
        base_dataset_kwargs: Optional[Dict[str, Any]] = None,
        document_converters: Optional[
            Dict[Union[Type[Document], str], Union[Callable[..., Document], Dict[str, str], str]]
        ] = None,
        **kwargs,
    ):
        self.base_builder = None
        config_name = kwargs.get("config_name", None)
        base_dataset_path = self.BASE_DATASET_PATHS.get(config_name, self.BASE_DATASET_PATH)
        if base_dataset_path is not None:
            base_dataset_kwargs = base_dataset_kwargs or {}
            base_builder_kwargs: Dict[str, Any] = {}

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
            self.base_builder = hf_datasets.load.load_dataset_builder(
                path=base_dataset_path,
                **base_builder_kwargs,
            )
            # Ensure that self and self.base_builder are derived from the same subclass of
            # hf_datasets.builder.DatasetBuilder.
            base_builder_general_parent_class = get_general_dataset_builder_parent_class(
                self.base_builder
            )
            self_general_parent_class = get_general_dataset_builder_parent_class(self)
            if base_builder_general_parent_class != self_general_parent_class:
                raise TypeError(
                    f"The PyTorch-IE dataset builder class '{type(self).__name__}' is derived from "
                    f"{self_general_parent_class}, but the base builder is not which is not allowed. The base builder "
                    f"is of type '{type(self.base_builder).__name__}' that is derived from "
                    f"{base_builder_general_parent_class}. Consider to derive your PyTorch-IE dataset builder "
                    f"'{type(self).__name__}' from a PyTorch-IE variant of "
                    f"'{base_builder_general_parent_class.__name__}'."
                )

            # append the base_builder config_id to the hash, otherwise the base_builder config arguments
            # are not respected in the cache fingerprint
            if "hash" in kwargs:
                kwargs["hash"] = f"{kwargs['hash']}-{self.base_builder.config_id}"

            # set base path to base builder base path. This is required so that the download manager
            # works correctly with relative paths.
            kwargs["base_path"] = self.base_builder.base_path

        super().__init__(**kwargs)

        self.document_converters = dict(self.DOCUMENT_CONVERTERS)
        if document_converters is not None:
            for document_type_or_str, document_converter_or_str in document_converters.items():
                document_type = resolve_target(document_type_or_str)
                if isinstance(document_type, type) and issubclass(document_type, Document):
                    document_converter: Union[Callable[..., Any], dict[str, str]]
                    if isinstance(document_converter_or_str, str):
                        document_converter = resolve_target(document_converter_or_str)
                    else:
                        document_converter = document_converter_or_str

                    self.document_converters[document_type] = document_converter
                else:
                    raise TypeError(
                        f"The key '{document_type_or_str}' for one of the converters "
                        f"can not be resolved to a document type."
                    )

    def _info(self):
        return self.base_builder._info()

    def _split_generators(self, dl_manager):
        return self.base_builder._split_generators(dl_manager)

    @property
    def document_type(self) -> Optional[Type[Document]]:
        return self.DOCUMENT_TYPES.get(self.config.name, self.DOCUMENT_TYPE)

    @abc.abstractmethod
    def _generate_document(self, example, **kwargs):
        pass

    def _generate_document_kwargs(self, dataset):
        return None

    @overload  # type: ignore
    def _convert_dataset_single(self, dataset: hf_datasets.IterableDataset) -> IterableDataset:
        ...

    @overload  # type: ignore
    def _convert_dataset_single(self, dataset: hf_datasets.Dataset) -> Dataset:
        ...

    def _convert_dataset_single(
        self, dataset: Union[hf_datasets.Dataset, hf_datasets.IterableDataset]
    ) -> Union[Dataset, IterableDataset]:
        document_type = self.document_type
        if document_type is None:
            raise TypeError(
                f"the builder has no DOCUMENT_TYPE or DOCUMENT_TYPES[{self.config.name}] defined"
            )

        fn = decorate_convert_to_dict_of_lists(self._generate_document)
        fn_kwargs = self._generate_document_kwargs(dataset)
        mapped_dataset = dataset.map(fn, fn_kwargs=fn_kwargs)
        dataset_type = get_pie_dataset_type(mapped_dataset)
        result = dataset_type.from_hf_dataset(
            dataset=mapped_dataset,
            document_type=document_type,
            document_converters=dict(self.document_converters),
        )
        return result

    @overload  # type: ignore
    def _convert_datasets(self, datasets: hf_datasets.DatasetDict) -> hf_datasets.DatasetDict:
        ...

    @overload  # type: ignore
    def _convert_datasets(
        self, datasets: hf_datasets.IterableDatasetDict
    ) -> hf_datasets.IterableDatasetDict:
        ...

    @overload  # type: ignore
    def _convert_datasets(self, datasets: hf_datasets.IterableDataset) -> IterableDataset:
        ...

    @overload  # type: ignore
    def _convert_datasets(self, datasets: hf_datasets.Dataset) -> Dataset:
        ...

    def _convert_datasets(
        self,
        datasets: Union[
            hf_datasets.Dataset,
            hf_datasets.IterableDataset,
            hf_datasets.DatasetDict,
            hf_datasets.IterableDatasetDict,
        ],
    ) -> Union[Dataset, IterableDataset, hf_datasets.DatasetDict, hf_datasets.IterableDatasetDict]:
        if isinstance(datasets, dict):
            return type(datasets)(
                {k: self._convert_dataset_single(v) for k, v in datasets.items()}
            )
        else:
            return self._convert_dataset_single(datasets)

    def as_dataset(
        self,
        split: Optional[hf_datasets.Split] = None,
        run_post_process=True,
        verification_mode: Optional[Union[hf_datasets.VerificationMode, str]] = None,
        ignore_verifications="deprecated",
        in_memory=False,
    ) -> Union[Dataset, hf_datasets.DatasetDict]:
        datasets = super().as_dataset(
            split=split,
            run_post_process=run_post_process,
            ignore_verifications=ignore_verifications,
            in_memory=in_memory,
            verification_mode=verification_mode,
        )
        converted_datasets = self._convert_datasets(datasets=datasets)
        return converted_datasets

    def as_streaming_dataset(
        self,
        split: Optional[str] = None,
        base_path: Optional[str] = None,
    ) -> Union[IterableDataset, hf_datasets.IterableDatasetDict]:  # type: ignore
        datasets: Union[
            hf_datasets.IterableDataset, hf_datasets.IterableDatasetDict
        ] = super().as_streaming_dataset(
            split=split, base_path=base_path
        )  # type: ignore
        converted_datasets = self._convert_datasets(datasets=datasets)
        return converted_datasets


class GeneratorBasedBuilder(PieDatasetBuilder, hf_datasets.builder.GeneratorBasedBuilder):
    def _generate_examples(self, *args, **kwargs):
        return self.base_builder._generate_examples(*args, **kwargs)


class ArrowBasedBuilder(PieDatasetBuilder, hf_datasets.builder.ArrowBasedBuilder):
    def _generate_tables(self, *args, **kwargs):
        return self.base_builder._generate_tables(*args, **kwargs)
