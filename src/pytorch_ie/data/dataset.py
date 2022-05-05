from functools import wraps
from typing import Callable, List, Optional, Type, Union

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
