import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, SupportsIndex, Type, Union

import datasets
from pytorch_ie.core import Document
from pytorch_ie.data.dataset import Dataset, IterableDataset
from pytorch_ie.utils.hydra import resolve_target

from .common import (
    EnterDatasetDictMixin,
    EnterDatasetMixin,
    ExitDatasetDictMixin,
    ExitDatasetMixin,
)

logger = logging.getLogger(__name__)


def get_pie_dataset_type(
    hf_dataset: Union[datasets.Dataset, datasets.IterableDataset]
) -> Union[Type[Dataset], Type[IterableDataset]]:
    if isinstance(hf_dataset, datasets.Dataset):
        return Dataset
    elif isinstance(hf_dataset, datasets.IterableDataset):
        return IterableDataset
    else:
        raise ValueError(
            f"dataset_split must be of type Dataset or IterableDataset, but is {type(hf_dataset)}"
        )


class DatasetDict(datasets.DatasetDict):
    def __getitem__(self, k) -> Union[Dataset, IterableDataset]:  # type: ignore
        """returns an individual dataset split"""

        dataset = super().__getitem__(k)
        if isinstance(dataset, (Dataset, IterableDataset)):
            return dataset
        else:
            raise TypeError(f"dataset must be of type Dataset, but is {type(dataset)}")

    @classmethod
    def load_dataset(cls, *args, **kwargs) -> "DatasetDict":
        return cls(datasets.load_dataset(*args, **kwargs))

    @classmethod
    def from_hf(
        cls,
        hf_dataset: Union[
            datasets.DatasetDict,
            datasets.IterableDatasetDict,
            datasets.Dataset,
            datasets.IterableDataset,
        ],
        document_type: Union[str, Type[Document]],
    ) -> "DatasetDict":
        """Creates a PIE DatasetDict from a HuggingFace DatasetDict, IterableDatasetDict, Dataset, or IterableDataset.
        If the input is a Dataset or IterableDataset, we create a DatasetDict with one split named "train".

        Args:
            hf_dataset: HuggingFace (Iterable)Dataset(Dict)
            document_type: document type of the dataset. Can be a subclass of Document or string that can be
                resolved to such a type.
        """

        doc_type = resolve_target(document_type)
        if not isinstance(doc_type, type) or not issubclass(doc_type, Document):
            raise TypeError(f"document_type must be a subclass of Document, but is {doc_type}")
        if isinstance(hf_dataset, (datasets.Dataset, datasets.IterableDataset)):
            hf_dataset = datasets.DatasetDict({"train": hf_dataset})
        res = cls(
            {
                k: get_pie_dataset_type(v).from_hf_dataset(v, document_type=doc_type)
                for k, v in hf_dataset.items()
            }
        )
        return res

    @classmethod
    def from_json(  # type: ignore
        cls,
        document_type: Union[Type[Document], str],
        **kwargs,
    ) -> "DatasetDict":
        """Creates a PIE DatasetDict from JSONLINE files. Uses `datasets.load_dataset("json")` under the hood.

        Args:
            document_type: document type of the dataset
            **kwargs: additional keyword arguments for `datasets.load_dataset()`
        """

        hf_dataset = datasets.load_dataset("json", **kwargs)
        if isinstance(
            hf_dataset,
            (
                datasets.DatasetDict,
                datasets.IterableDatasetDict,
                datasets.Dataset,
                datasets.IterableDataset,
            ),
        ):
            return cls.from_hf(hf_dataset, document_type=document_type)
        else:
            raise TypeError(
                f"expected datasets.DatasetDict, datasets.IterableDatasetDict, datasets.Dataset, "
                f"or datasets.IterableDataset, but got {type(hf_dataset)}"
            )

    def to_json(self, path: Union[str, Path], **kwargs) -> None:
        """Serializes the DatasetDict. We convert all documents with `.asdict()`
        and dump them with `json.dump()` to one JSONLINE file per split.

        Args:
            path: path to the output directory
            **kwargs: additional keyword arguments for `json.dump()`
        """

        path = Path(path)
        for split, dataset in self.items():
            split_path = path / split
            logger.info(f'serialize documents to "{split_path}" ...')
            os.makedirs(split_path, exist_ok=True)
            file_name = split_path / "documents.jsonl"
            with open(file_name, "w") as f:
                for doc in dataset:
                    f.write(json.dumps(doc.asdict(), **kwargs) + "\n")

    @property
    def document_type(self) -> Type[Document]:
        """Returns the document type of the dataset splits.

        Raises an error if there are no splits in the dataset or if the dataset splits have different
        document types.
        """

        if len(self) == 0:
            raise ValueError("dataset does not contain any splits, cannot determine document type")
        document_types = {ds.document_type for ds in self.values()}
        if len(document_types) > 1:
            raise ValueError(
                f"dataset contains splits with different document types: {document_types}"
            )
        return next(iter(document_types))

    @property
    def dataset_type(self) -> Union[Type[Dataset], Type[IterableDataset]]:
        """Returns the dataset type of the dataset splits, i.e. either `Dataset` or `IterableDataset`.

        Raises an error if there are no splits in the dataset or if the dataset splits have different
        dataset types.
        """

        if len(self) == 0:
            raise ValueError(
                "dataset does not contain any splits, cannot determine the dataset type"
            )
        dataset_types = {type(ds) for ds in self.values()}
        if len(dataset_types) > 1:
            raise ValueError(
                f"dataset contains splits with different dataset types: {dataset_types}"
            )
        return next(iter(dataset_types))

    def map(  # type: ignore
        self,
        function: Optional[Union[Callable, str]] = None,
        result_document_type: Optional[Union[str, Type[Document]]] = None,
        **kwargs,
    ) -> "DatasetDict":
        """Applies a function to all documents in the dataset.

        If the function is an object and is derived from the following mixins, the respective logic
        is applied:
        - EnterDatasetMixin: `enter_dataset(dataset_split, split_name)` is called before the function is
            applied to a dataset split
        - ExitDatasetMixin: `exit_dataset(processed_dataset_split, split_name)` is called after the function
            is applied to a dataset split
        - EnterDatasetDictMixin: `enter_dataset_dict(dataset_dict)` is called before any dataset split is
            processed (and before any `enter_dataset()` is called)
        - ExitDatasetDictMixin: `exit_dataset_dict(processed_dataset_dict)` is called after all dataset splits
            are processed (and after all `exit_dataset()` are called)

        Args:
            function: function to apply to the documents. If `None`, the identity function is used. If `str`,
                the function is resolved from the global namespace.
            result_document_type: optional document type of the resulting dataset. Can be a subclass of Document or
                string that can be resolved to such a type. If not provided, it is tried to infer it from the
                function signature. If this is not possible, the document type of the input dataset
                is used.
            **kwargs: additional keyword arguments for `datasets.Dataset.map()`
        """

        if function is not None:
            func = resolve_target(function)
            if not callable(func):
                raise TypeError(f"function must be callable, but is of type {type(func)}")
        else:

            def identity(x):
                # exclude from coverage because its usage happens in the map which is not collected
                return x  # pragma: no cover

            func = identity
        map_kwargs = dict(function=func, **kwargs)
        if result_document_type is not None:
            map_kwargs["result_document_type"] = resolve_target(result_document_type)

        if isinstance(func, EnterDatasetDictMixin):
            func.enter_dataset_dict(self)

        result_dict = {}
        for split, dataset in self.items():
            if isinstance(func, EnterDatasetMixin):
                func.enter_dataset(dataset=dataset, name=split)
            result_dict[split] = dataset.map(**map_kwargs)
            if isinstance(func, ExitDatasetMixin):
                func.exit_dataset(dataset=result_dict[split], name=split)

        result = type(self)(result_dict)

        if isinstance(func, ExitDatasetDictMixin):
            func.exit_dataset_dict(result)

        return result

    def select(
        self,
        split: str,
        start: Optional[SupportsIndex] = None,
        stop: Optional[SupportsIndex] = None,
        step: Optional[SupportsIndex] = None,
        **kwargs,
    ) -> "DatasetDict":
        """Reduce a certain dataset split to a selection of its documents. This is similar to the Huggingface
        `select()`, but adds optional parameters `start`, `stop`, `step` that will be used to create indices,
        if available.

        Args:
            split: name of the dataset split to modify
            start: optional start index of the selection
            stop: optional stop index of the selection
            step: optional step size of the selection
            **kwargs: additional keyword arguments for `datasets.Dataset.select()`
        """

        if stop is not None:
            range_args = [stop]
            if start is not None:
                range_args = [start] + range_args
            if step is not None:
                range_args = range_args + [step]
            kwargs["indices"] = range(*range_args)

        if "indices" in kwargs:
            result = type(self)(self)
            pie_split = result[split]
            if not isinstance(pie_split, Dataset):
                raise TypeError(
                    f"can only select from a Dataset, but the split '{split}' is of type {type(pie_split)}"
                )
            result[split] = Dataset.from_hf_dataset(
                dataset=pie_split.select(**kwargs), document_type=pie_split.document_type
            )
            return result
        else:
            if len(kwargs) > 0:
                logger.warning(
                    f"arguments for dataset.select() available, but they do not contain 'indices' which is required, "
                    f"so we do not call select. provided arguments: \n{json.dumps(kwargs, indent=2)}"
                )
            return self

    def rename_splits(
        self,
        mapping: Optional[Dict[str, str]] = None,
        keep_other_splits: bool = True,
    ) -> "DatasetDict":
        """Renames the dataset splits.

        Args:
            mapping: mapping from old split names to new split names.
            keep_other_splits: if `True` (default), splits not contained in `mapping` are kept in the dataset
        """

        if mapping is None:
            mapping = {}
        result = type(self)(
            {
                mapping.get(name, name): data
                for name, data in self.items()
                if name in mapping or keep_other_splits
            }
        )
        return result

    def add_test_split(
        self,
        source_split: str = "train",
        target_split: str = "test",
        **kwargs,
    ) -> "DatasetDict":
        """Adds a test split to the dataset by splitting the source split. Uses the Huggingface
        `train_test_split()` method."""

        pie_split = self[source_split]
        if not isinstance(pie_split, Dataset):
            raise TypeError(
                f"can only create a train-test-split from a Dataset, but the source split '{source_split}' is of type "
                f"{type(pie_split)}"
            )
        split_result_hf = pie_split.train_test_split(**kwargs)
        split_result = type(self)(
            {
                name: Dataset.from_hf_dataset(ds, document_type=pie_split.document_type)
                for name, ds in split_result_hf.items()
            }
        )
        res = type(self)(self)
        res[source_split] = split_result["train"]
        res[target_split] = split_result["test"]
        split_sizes = {k: len(v) for k, v in res.items()}
        logger.info(f"dataset size after adding the split: {split_sizes}")
        return res

    def drop_splits(self, split_names: List[str]) -> "DatasetDict":
        """Drops splits from the dataset.

        Args:
            split_names: names of the splits to drop
        """

        result = type(self)({name: ds for name, ds in self.items() if name not in split_names})
        return result

    def concat_splits(self, splits: List[str], target: str) -> "DatasetDict":
        """Concatenates selected splits into a new split.

        Args:
            splits: names of the splits to concatenate
            target: name of the new split
        """

        if any(split not in self for split in splits):
            raise ValueError(
                f"not all splits to concatenate are present in the dataset: {splits}, {self.keys()}"
            )
        if len(splits) == 0:
            raise ValueError("please provide at least one split to concatenate")
        result = type(self)({name: ds for name, ds in self.items() if name not in splits})
        if not issubclass(self.dataset_type, Dataset):
            raise TypeError(
                f"can only concatenate splits if the dataset type is a Dataset, but it is {self.dataset_type}"
            )
        splits_to_concat: List[Dataset] = [self[name] for name in splits]  # type: ignore
        if any(self.dataset_type != type(ds) for ds in splits_to_concat):
            raise ValueError(
                f"not all splits to concatenate have the same dataset type: "
                f"{({name: type(self[name]) for name in splits})}"
            )
        # TODO: why do we need to ignore the typing here?
        concatenated = datasets.concatenate_datasets(splits_to_concat)  # type: ignore
        if not issubclass(self.dataset_type, type(concatenated)):
            raise ValueError(
                f"concatenated dataset is not of the same type as the original dataset: "
                f"{self.dataset_type}, {type(concatenated)}"
            )
        result[target] = self.dataset_type.from_hf_dataset(
            concatenated, document_type=self.document_type
        )
        split_sizes = {k: len(v) for k, v in result.items()}
        logger.info(f"dataset size after concatenating splits: {split_sizes}")
        return result

    def filter(  # type: ignore
        self,
        split: str,
        function: Optional[Union[Callable[[Dict], bool], str]] = None,
        result_split_name: Optional[str] = None,
        **kwargs,
    ) -> "DatasetDict":
        """Filters a dataset split using a filter function.

        Note: In contrast to `map`, the filter function gets the example dict instead of a document as input
        because the PIE variant of `Dataset.filter()` is not yet implemented and, thus, the Huggingface
        variant is internally used instead.

        Args:
            split: name of the split to filter
            function: filter function that is called on each example dict. Can be provided as a callable or as a
                string that is resolved to a callable using `resolve_target()`.
            result_split_name: name of the split to store the filtered examples in. If `None`, the filtered examples
                are stored in the same split as the original examples.

        """

        if function is not None:
            # create a shallow copy to not modify the input
            result = type(self)(self)
            function = resolve_target(function)
            pie_split = result[split]
            # TODO: Implement pytorch_ie.Dataset.filter() in a similar way such as map() to make use of the
            #  document type. For now, the filter function is called directly on the HF dataset and thus needs to
            #  accept a dict as input.
            # we need to convert the dataset back to HF because the filter function internally uses map() which will
            # break if the PIE variant is used
            hf_split: Union[datasets.Dataset, datasets.IterableDataset]
            if isinstance(pie_split, Dataset):
                hf_split = datasets.Dataset(**Dataset.get_base_kwargs(pie_split))
            elif isinstance(pie_split, IterableDataset):
                hf_split = datasets.IterableDataset(**IterableDataset.get_base_kwargs(pie_split))
            else:
                raise ValueError(f"dataset split has unknown type: {type(pie_split)}")
            hf_split_filtered = hf_split.filter(function=function, **kwargs)
            target_split_name = result_split_name or split
            target_split = type(pie_split).from_hf_dataset(
                dataset=hf_split_filtered,  # type: ignore
                document_type=pie_split.document_type,
            )
            # iterable datasets do not have a length
            if not isinstance(target_split, IterableDataset):
                logger.info(
                    f"filtered split [{target_split_name}] has {len(target_split)} entries"
                )
            result[target_split_name] = target_split
            return result
        else:
            return self

    def move_to_new_split(
        self,
        ids: Optional[List[str]] = None,
        filter_function: Optional[Union[Callable[[Dict[str, Any]], bool], str]] = None,
        source_split: str = "train",
        target_split: str = "test",
    ) -> "DatasetDict":
        """Moves examples from one split to another split. ids or a filter function can be provided to select the
        examples to move.

        Args:
            ids: list of ids of the examples to move
            filter_function: filter function that is called on each example dict. Can be provided as a callable or as a
                string that can be resolved to such a callable.
            source_split: name of the split to move the examples from
            target_split: name of the split to move the examples to
        """

        if filter_function is not None:
            filter_func = resolve_target(filter_function)
        else:
            if ids is None:
                raise ValueError("please provide either a list of ids or a filter function")

            ids_set = set(ids)

            def filter_with_ids(ex: Dict[str, Any]):
                # exclude from coverage because its usage happens in the map which is not collected
                return ex["id"] in ids_set  # pragma: no cover

            filter_func = filter_with_ids

        dataset_with_only_ids = self.filter(
            split=source_split,
            function=filter_func,
        )
        dataset_without_ids = self.filter(
            split=source_split,
            function=lambda ex: not filter_func(ex),
        )
        dataset_without_ids[target_split] = dataset_with_only_ids[source_split]

        split_sizes = {k: len(v) for k, v in dataset_without_ids.items()}
        logger.info(f"dataset size after moving to new split: {split_sizes}")
        return dataset_without_ids

    def cast_document_type(
        self, new_document_type: Union[Type[Document], str], **kwargs
    ) -> "DatasetDict":
        """Casts the document type of all splits to a new document type."""

        new_type = resolve_target(new_document_type)

        result = type(self)(
            {
                name: ds.cast_document_type(new_document_type=new_type, **kwargs)
                for name, ds in self.items()
            }
        )
        return result
