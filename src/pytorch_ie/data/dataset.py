import logging
from functools import wraps
from inspect import Signature, isclass, signature
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

import datasets
import pandas as pd
from datasets.formatting import _register_formatter

from pytorch_ie.core.document import Document
from pytorch_ie.data.dataset_formatter import DocumentFormatter

logger = logging.getLogger(__name__)

_register_formatter(DocumentFormatter, "document")


def decorate_convert_to_dict_of_lists(f):
    """
    Decorate the mapped function, so that converts a single Document to a dict,
    and a list of Documents into a dict of lists.
    """

    @wraps(f)
    def decorated(item, *args, **kwargs):
        if isinstance(item, list):
            # Convert a list of dicts into a dict of lists.
            return pd.DataFrame([e.asdict() for e in f(item, *args, **kwargs)]).to_dict(
                orient="list"
            )
        else:
            return f(item, *args, **kwargs).asdict()

    return decorated


E = TypeVar("E")


def dl_to_ld(dict_list: Dict[str, List[E]]) -> List[Dict[str, E]]:
    # Convert a dict of lists to a list of dicts
    return [dict(zip(dict_list, t)) for t in zip(*dict_list.values())]


def ld_to_dl(
    list_dict: List[Dict[str, E]], keys: Optional[Iterable[str]] = None
) -> Dict[str, List[E]]:
    # Convert a list of dicts to a dict of lists.
    # Provide keys to create the expected format when lists are empty.
    if keys is None:
        keys = list_dict[0]
    return {k: [dic[k] for dic in list_dict] for k in keys}


def decorate_convert_to_document_and_back(f, document_type: Type[Document], batched: bool):
    @wraps(f)
    def decorated(item, *args, **kwargs):
        if batched:
            # Convert a list of dicts into a dict of lists.
            return ld_to_dl(
                [
                    e.asdict()
                    for e in f(
                        [document_type.fromdict(x) for x in dl_to_ld(item)], *args, **kwargs
                    )
                ],
                # passing the keys allows to work correctly with empty lists
                keys=item.keys(),
            )
        else:
            return f(document_type.fromdict(item), *args, **kwargs).asdict()

    return decorated


def _check_fields_for_casting(
    field_mapping: Dict[str, str],
    current_document_type: Type[Document],
    new_document_type: Type[Document],
    column_names: list[str],
) -> Tuple[Set[str], Set[str]]:
    original_fields = {field.name: field for field in current_document_type.fields()}
    new_fields = {field.name: field for field in new_document_type.fields()}
    hidden_fields = set(column_names) - set(original_fields)
    fields_to_map_not_in_original_fields = (
        set(field_mapping) - set(original_fields) - set(hidden_fields)
    )
    if len(fields_to_map_not_in_original_fields) > 0:
        raise ValueError(
            f"some fields to rename are not in the original document_type or hidden fields: "
            f"{fields_to_map_not_in_original_fields}"
        )
    mapped_but_not_in_new_fields = set(field_mapping.values()) - set(new_fields)
    if len(mapped_but_not_in_new_fields) > 0:
        raise ValueError(
            f"some renamed fields are not in the new document_type: {mapped_but_not_in_new_fields}"
        )
    original_fields_mapped = {
        field_mapping.get(f_name, f_name): f for f_name, f in original_fields.items()
    }
    added_field_names = set(new_fields) - set(original_fields_mapped)
    removed_field_names = set(original_fields) - set(new_fields) - set(field_mapping)

    # Sanity checks
    kept_field_names = set(original_fields_mapped) & set(new_fields)
    for f_name_mapped in kept_field_names:
        f = original_fields_mapped[f_name_mapped]
        new_f = new_fields[f_name_mapped]
        if not (
            f.type == new_f.type
            and f.default == new_f.default
            and f.default_factory == new_f.default_factory
        ):
            raise ValueError(f"new field is not the same as old field:\n{new_f}\nvs\n{f}")

    return removed_field_names, added_field_names


def _infer_document_type_from_function_return(
    function: Callable, strict: bool = True
) -> Optional[Type[Document]]:
    # try to infer the document type from the return type annotation of function
    return_signature = signature(function).return_annotation
    if not return_signature == Signature.empty:
        if not isclass(return_signature) or not issubclass(return_signature, Document):
            if strict:
                raise TypeError(
                    f"the return type annotation of the function used with map is not a subclass of Document"
                )
            else:
                logger.warning(
                    f"the return type annotation of the function used with map is not a subclass of Document"
                )
                return None
        return return_signature
    return None


D = TypeVar("D", bound=Document)
DocumentConvertersType = Dict[Type[D], Union[Callable[..., D], Dict[str, str]]]


def _get_best_dataset_converter_with_types(
    dataset: Union["IterableDataset", "Dataset"],
    document_type: Union[Type[Document]],
) -> Tuple[Union[Callable[..., Document], Dict[str, str]], Type[Document], Type[Document]]:
    # first try to find an exact match
    if document_type in dataset.document_converters:
        return dataset.document_converters[document_type], document_type, document_type

    # then try to find a match with a superclass
    for registered_dt, candidate_converter in dataset.document_converters.items():
        if issubclass(registered_dt, document_type):
            return candidate_converter, document_type, registered_dt

    # then try to find a match with a subclass
    for registered_dt, candidate_converter in dataset.document_converters.items():
        if issubclass(document_type, registered_dt):
            return candidate_converter, document_type, registered_dt

    raise ValueError(
        f"No valid key (either subclass or superclass) was found for the document type '{document_type}' "
        f"in the document_converters of the dataset. Available keys: {set(dataset.document_converters)}. "
        f"Consider adding a respective converter to the dataset with "
        f"dataset.register_document_converter(my_converter_method) where my_converter_method should accept "
        f"{dataset.document_type} as input and return '{document_type}'."
    )


@overload
def dataset_to_document_type(
    dataset: "Dataset",
    document_type: Type[Document],
    **kwargs,
) -> "Dataset":
    ...


@overload
def dataset_to_document_type(
    dataset: "IterableDataset",
    document_type: Type[Document],
    **kwargs,
) -> "IterableDataset":
    ...


def dataset_to_document_type(
    dataset: Union["IterableDataset", "Dataset"],
    document_type: Type[Document],
    **kwargs,
) -> Union["IterableDataset", "Dataset"]:

    # do nothing if the document type is already the requested type
    if document_type == dataset.document_type:
        logger.info(f"The dataset has already the requested document type {document_type}.")
        return dataset

    converter, requested_type, registered_type = _get_best_dataset_converter_with_types(
        dataset=dataset,
        document_type=document_type,
    )

    result = dataset
    if callable(converter):
        result = result.map(
            function=converter,
            result_document_type=registered_type,
            fn_kwargs=kwargs,
        )
    else:
        result = result.cast_document_type(
            new_document_type=registered_type, field_mapping=converter, **kwargs
        )
    # if the type is not the same or a subclass of the requested type, try to cast (again)
    if not issubclass(registered_type, requested_type):
        result = result.cast_document_type(new_document_type=requested_type)

    # remove the document converters because they are not valid anymore
    result.document_converters = {}

    return result


def dataset_register_document_converter(
    dataset: Union["Dataset", "IterableDataset"],
    converter: Union[Callable[..., D], Dict[str, str]],
    document_type: Optional[Type[D]] = None,
) -> None:
    if callable(converter) and document_type is None:
        dt = _infer_document_type_from_function_return(converter)
    else:
        dt = document_type
    if not (isinstance(dt, type) and issubclass(dt, Document)):
        raise TypeError(
            f"the (inferred) document_type {dt} is not a subclass of Document. "
            "Please provide a document_type or a converter with a return type annotation."
        )
    dataset.document_converters[dt] = converter


class Dataset(datasets.Dataset):
    def __init__(
        self,
        document_type: Type[Document],
        arrow_table: datasets.table.Table,
        info: Optional[datasets.DatasetInfo] = None,
        split: Optional[datasets.NamedSplit] = None,
        indices_table: Optional[datasets.table.Table] = None,
        fingerprint: Optional[str] = None,
        document_converters: Optional[DocumentConvertersType] = None,
    ):
        super().__init__(
            arrow_table=arrow_table,
            info=info,
            split=split,
            indices_table=indices_table,
            fingerprint=fingerprint,
        )

        self.document_type = document_type
        self.set_format("document", document_type=document_type)
        self.document_converters = document_converters or {}

    @classmethod
    def get_base_kwargs(cls, dataset: datasets.Dataset):
        return dict(
            arrow_table=dataset._data,
            info=dataset.info,
            split=dataset.split,
            indices_table=dataset._indices,
            fingerprint=dataset._fingerprint,
        )

    @classmethod
    def from_hf_dataset(
        cls,
        dataset: datasets.Dataset,
        document_type: Type[Document],
        document_converters: Optional[DocumentConvertersType] = None,
    ) -> "Dataset":
        document_dataset = cls(
            document_type=document_type,
            document_converters=document_converters,
            **cls.get_base_kwargs(dataset),
        )
        return document_dataset

    def apply_hf_func(self, func, **kwargs) -> "Dataset":
        return Dataset.from_hf_dataset(
            func(self, **kwargs),
            document_type=self.document_type,
            document_converters=self.document_converters,
        )

    def register_document_converter(
        self,
        converter: Union[Callable[..., D], Dict[str, str]],
        document_type: Optional[Type[D]] = None,
    ) -> None:
        dataset_register_document_converter(
            dataset=self,
            converter=converter,
            document_type=document_type,
        )

    def to_document_type(
        self,
        document_type: Type[Document],
        **kwargs,
    ) -> "Dataset":
        return dataset_to_document_type(
            dataset=self,
            document_type=document_type,
            **kwargs,
        )

    def map(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        with_rank: bool = False,
        input_columns: Optional[Union[str, List[str]]] = None,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        drop_last_batch: bool = False,
        remove_columns: Optional[Union[str, List[str]]] = None,
        keep_in_memory: bool = False,
        load_from_cache_file: Optional[bool] = None,
        cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        features: Optional[datasets.Features] = None,
        disable_nullable: bool = False,
        fn_kwargs: Optional[dict] = None,
        num_proc: Optional[int] = None,
        suffix_template: str = "_{rank:05d}_of_{num_proc:05d}",
        new_fingerprint: Optional[str] = None,
        desc: Optional[str] = None,
        as_documents: bool = True,
        result_document_type: Optional[Type[Document]] = None,
    ) -> "Dataset":
        dataset = super().map(
            function=decorate_convert_to_dict_of_lists(function) if as_documents else function,
            with_indices=with_indices,
            with_rank=with_rank,
            input_columns=input_columns,
            batched=batched,
            batch_size=batch_size,
            drop_last_batch=drop_last_batch,
            remove_columns=remove_columns,
            keep_in_memory=keep_in_memory,
            # ignore typing because typing in Huggingface Dataset.map() is incorrect
            load_from_cache_file=load_from_cache_file,  # type: ignore
            cache_file_name=cache_file_name,
            writer_batch_size=writer_batch_size,
            features=features,
            disable_nullable=disable_nullable,
            fn_kwargs=fn_kwargs,
            num_proc=num_proc,
            suffix_template=suffix_template,
            new_fingerprint=new_fingerprint,
            desc=desc,
        )

        if result_document_type is None:
            result_document_type = self.document_type

        return Dataset.from_hf_dataset(
            dataset,
            document_type=result_document_type,
            document_converters=self.document_converters,
        )

    def cast_document_type(
        self,
        new_document_type: Type[D],
        remove_columns: bool = False,
        field_mapping: Optional[Dict[str, str]] = None,
    ) -> "Dataset":
        field_mapping = field_mapping or {}

        removed_field_names, added_field_names = _check_fields_for_casting(
            field_mapping=field_mapping,
            current_document_type=self.document_type,
            new_document_type=new_document_type,
            column_names=self.column_names,
        )

        new_hf_dataset = datasets.Dataset(**self.get_base_kwargs(self))

        if remove_columns:
            new_hf_dataset = new_hf_dataset.remove_columns(list(removed_field_names))

        rename_targets_already_in_columns = (
            set(field_mapping.values()) - set(field_mapping)
        ) & set(new_hf_dataset.column_names)
        if len(rename_targets_already_in_columns) > 0:
            raise ValueError(
                f"rename targets are already in column names: {rename_targets_already_in_columns}. Did you miss "
                f"to set remove_columns=True in a previous call of cast_document_type?"
            )

        new_hf_dataset = new_hf_dataset.rename_columns(field_mapping)
        for f_name in added_field_names:
            if f_name not in new_hf_dataset.column_names:
                # add empty columns
                new_hf_dataset = new_hf_dataset.add_column(
                    name=f_name, column=len(new_hf_dataset) * [{}]
                )
        new_dataset = Dataset.from_hf_dataset(
            new_hf_dataset,
            document_type=new_document_type,
            document_converters=self.document_converters,
        )

        return new_dataset


class IterableDataset(datasets.IterableDataset):
    def __init__(
        self,
        document_type: Type[Document],
        hidden_columns: Optional[Set[str]] = None,
        document_converters: Optional[DocumentConvertersType] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.document_type = document_type
        self._document_field_names = [field.name for field in document_type.fields()]
        self.hidden_columns = set()
        if hidden_columns is not None:
            self.hidden_columns.update(hidden_columns)
        self.document_converters = document_converters or {}

    @property
    def column_names(self) -> List[str]:
        return self._document_field_names + list(self.hidden_columns)

    @classmethod
    def get_base_kwargs(cls, dataset: datasets.IterableDataset):
        return dict(
            ex_iterable=dataset._ex_iterable,
            info=dataset.info,
            split=dataset.split,
            formatting=dataset._formatting,
            shuffling=dataset._shuffling,
            distributed=dataset._distributed,
            token_per_repo_id=dataset._token_per_repo_id,
        )

    @classmethod
    def from_hf_dataset(
        cls,
        dataset: datasets.IterableDataset,
        document_type: Type[Document],
        hidden_columns: Optional[Set[str]] = None,
        document_converters: Optional[DocumentConvertersType] = None,
    ) -> "IterableDataset":
        dataset = cls(
            document_type=document_type,
            hidden_columns=hidden_columns,
            document_converters=document_converters,
            **cls.get_base_kwargs(dataset),
        )
        return dataset

    def __iter__(self):
        for example in iter(super().__iter__()):
            yield self.document_type.fromdict(example)

    def register_document_converter(
        self,
        converter: Union[Callable[..., D], Dict[str, str]],
        document_type: Optional[Type[D]] = None,
    ) -> None:
        dataset_register_document_converter(
            dataset=self,
            converter=converter,
            document_type=document_type,
        )

    def to_document_type(
        self,
        document_type: Type[Document],
        **kwargs,
    ) -> "IterableDataset":
        return dataset_to_document_type(
            dataset=self,
            document_type=document_type,
            **kwargs,
        )

    def map(  # type: ignore
        self,
        function: Optional[Callable] = None,
        batched: bool = False,
        as_documents: bool = True,
        result_document_type: Optional[Type[Document]] = None,
        **kwargs,
    ) -> "IterableDataset":
        dataset_mapped = super().map(
            function=decorate_convert_to_document_and_back(
                function, document_type=self.document_type, batched=batched
            )
            if as_documents
            else function,
            batched=batched,
            **kwargs,
        )

        if result_document_type is None:
            result_document_type = self.document_type

        return IterableDataset.from_hf_dataset(
            dataset_mapped,
            document_type=result_document_type,
            document_converters=self.document_converters,
        )

    def apply_hf_func(self, func, **kwargs) -> "IterableDataset":
        return IterableDataset.from_hf_dataset(
            func(self, **kwargs),
            document_type=self.document_type,
            hidden_columns=self.hidden_columns,
            document_converters=self.document_converters,
        )

    def cast_document_type(
        self,
        new_document_type: Type[D],
        remove_columns: bool = False,
        field_mapping: Optional[Dict[str, str]] = None,
    ) -> "IterableDataset":
        field_mapping = field_mapping or {}

        removed_field_names, added_field_names = _check_fields_for_casting(
            field_mapping=field_mapping,
            current_document_type=self.document_type,
            new_document_type=new_document_type,
            column_names=self.column_names,
        )
        hidden_columns = set(self.hidden_columns)
        new_hf_dataset = datasets.IterableDataset(**self.get_base_kwargs(self))

        if remove_columns:
            new_hf_dataset = new_hf_dataset.remove_columns(column_names=list(removed_field_names))
        else:
            hidden_columns.update(removed_field_names)

        rename_targets_already_in_columns = (
            set(field_mapping.values()) - set(field_mapping)
        ) & hidden_columns
        if len(rename_targets_already_in_columns) > 0:
            raise ValueError(
                f"rename targets are already in column names: {rename_targets_already_in_columns}. Did you "
                f"miss to set remove_columns=True in a previous call of cast_document_type?"
            )

        new_hf_dataset = new_hf_dataset.rename_columns(column_mapping=field_mapping)

        new_dataset = IterableDataset.from_hf_dataset(
            new_hf_dataset,
            hidden_columns=hidden_columns,
            document_type=new_document_type,
            document_converters=self.document_converters,
        )

        return new_dataset

    def take(self, n) -> "IterableDataset":
        return self.apply_hf_func(datasets.IterableDataset.take, n=n)


def get_pie_dataset_type(
    hf_dataset: Union[datasets.Dataset, datasets.IterableDataset]
) -> Union[Type[Dataset], Type[IterableDataset]]:
    if isinstance(hf_dataset, datasets.Dataset):
        return Dataset
    elif isinstance(hf_dataset, datasets.IterableDataset):
        return IterableDataset
    else:
        raise TypeError(
            f"the dataset must be of type Dataset or IterableDataset, but is of type {type(hf_dataset)}"
        )
