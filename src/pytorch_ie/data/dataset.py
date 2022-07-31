from dataclasses import fields
from functools import wraps
from typing import Callable, Dict, List, Optional, Type, TypeVar, Union

import pandas as pd
from datasets.formatting import _register_formatter

import datasets
from pytorch_ie.core.document import Document
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
        new_document_type: D,
        allow_field_removal: bool = False,
        field_mapping: Optional[Dict[str, str]] = None,
    ) -> "Dataset":

        field_mapping = field_mapping or {}

        # check for consistency (and collect remove_fields)
        original_fields = {field.name: field for field in fields(self.document_type)}
        new_fields = {field.name: field for field in fields(new_document_type)}
        remove_fields = []
        for f_name, f in original_fields.items():
            f_name_mapped = field_mapping.get(f_name, f_name)
            if f_name_mapped not in new_fields:
                if allow_field_removal:
                    remove_fields.append(f_name)
                    continue
                raise ValueError(
                    f'field "{f_name}" of original document_type [{str(self.document_type)}] is missing from new '
                    f"document type [{str(new_document_type)}]"
                )

            new_f = new_fields[f_name_mapped]
            if not (
                f.type == new_f.type
                and f.metadata == new_f.metadata
                and f.default == new_f.default
                and f.default_factory == new_f.default_factory
            ):
                raise ValueError(f"new field is not the same as old field:\n{new_f}\nvs\n{f}")

        # def document_as_type(doc: Document, new_type: Type[D], field_mapping: Dict[str, str]) -> D:
        #    new_doc = new_type.fromdict({field_mapping.get(k, k): v for k, v in doc.asdict().items()})
        #    return new_doc

        mapped_dataset = self.map(
            Document.as_type,
            fn_kwargs=dict(new_type=new_document_type, field_mapping=field_mapping),
            # remove entries from "remove_fields" and also mapped fields that are not in the mapping result
            remove_columns=remove_fields
            + list(set(field_mapping.keys()) - set(field_mapping.values())),
        )
        new_dataset = Dataset.from_hf_dataset(
            dataset=mapped_dataset, document_type=new_document_type
        )
        return new_dataset
