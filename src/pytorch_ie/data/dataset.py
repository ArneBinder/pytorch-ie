from dataclasses import fields
from functools import wraps
from typing import Callable, Dict, List, Optional, Type, TypeVar, Union

import pandas as pd
from datasets.formatting import _register_formatter

import datasets
from pytorch_ie.core.document import Document, _get_annotation_fields
from pytorch_ie.data.dataset_formatter import DocumentFormatter

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


D = TypeVar("D", bound=Document)


class Dataset(datasets.Dataset):
    def __init__(
        self,
        document_type: Type[Document],
        arrow_table: datasets.table.Table,
        info: Optional[datasets.DatasetInfo] = None,
        split: Optional[datasets.NamedSplit] = None,
        indices_table: Optional[datasets.table.Table] = None,
        fingerprint: Optional[str] = None,
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

    @classmethod
    def from_hf_dataset(cls, dataset: datasets.Dataset, document_type):
        document_dataset = cls(
            document_type=document_type,
            arrow_table=dataset._data,
            info=dataset.info,
            split=dataset.split,
            indices_table=dataset._indices,
            fingerprint=dataset._fingerprint,
        )
        return document_dataset

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
        load_from_cache_file: bool = None,
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
            load_from_cache_file=load_from_cache_file,
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

        return Dataset.from_hf_dataset(dataset, document_type=self.document_type)

    def cast_document_type(
        self,
        new_document_type: Type[D],
        remove_columns: bool = False,
        field_mapping: Optional[Dict[str, str]] = None,
    ) -> "Dataset":

        field_mapping = field_mapping or {}

        original_fields = {
            field.name: field for field in _get_annotation_fields(list(fields(self.document_type)))
        }
        new_fields = {
            field.name: field for field in _get_annotation_fields(list(fields(new_document_type)))
        }
        hidden_fields = set(self.column_names) - set(original_fields)
        fields_to_map_not_in_original_fields = (
            set(field_mapping) - set(original_fields) - set(hidden_fields)
        )
        if len(fields_to_map_not_in_original_fields) > 0:
            raise ValueError(
                f"some fields to rename are not in the original document_type or hidden fields: {fields_to_map_not_in_original_fields}"
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
                and f.metadata == new_f.metadata
                and f.default == new_f.default
                and f.default_factory == new_f.default_factory
            ):
                raise ValueError(f"new field is not the same as old field:\n{new_f}\nvs\n{f}")

        new_hf_dataset = datasets.Dataset(
            arrow_table=self._data,
            info=self.info,
            split=self.split,
            indices_table=self._indices,
            fingerprint=self._fingerprint,
        )
        if remove_columns:
            new_hf_dataset = new_hf_dataset.remove_columns(list(removed_field_names))

        rename_targets_already_in_columns = (
            set(field_mapping.values()) - set(field_mapping)
        ) & set(new_hf_dataset.column_names)
        if len(rename_targets_already_in_columns) > 0:
            raise ValueError(
                f"rename targets are already in column names: {rename_targets_already_in_columns}. Did you miss to set remove_columns=True in a previous call of cast_document_type?"
            )

        new_hf_dataset = new_hf_dataset.rename_columns(field_mapping)
        for f_name in added_field_names:
            if f_name not in new_hf_dataset.column_names:
                # add empty columns
                new_hf_dataset = new_hf_dataset.add_column(
                    name=f_name, column=len(new_hf_dataset) * [{}]
                )
        new_dataset = Dataset.from_hf_dataset(new_hf_dataset, document_type=new_document_type)

        return new_dataset
