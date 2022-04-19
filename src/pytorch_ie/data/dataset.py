import datasets
from datasets.formatting import _register_formatter
from pytorch_ie.data.dataset_formatter import DocumentFormatter
from typing import Optional, Callable, Union, List
from functools import wraps


_register_formatter(DocumentFormatter, "document")


class Dataset(datasets.Dataset):
    @classmethod
    def from_hf_dataset(cls, dataset: datasets.Dataset):
        return cls(
            arrow_table=dataset._data,
            info=dataset.info,
            split=dataset.split,
            indices_table=dataset._indices,
            fingerprint=dataset._fingerprint,
        )

    def map(
        self,
        function: Optional[Callable] = None,
        as_documents: bool = False,
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
    ) -> "Dataset":
        def decorate(f):
            """
            Decorate the mapped function, so that its first argument is wrapped with a LazyDict to be used internally
            but a standard dictionary is returned at the end of the mapping.
            """
            import pandas as pd

            @wraps(f)
            def decorated(item, *args, **kwargs):
                if isinstance(item, list):
                    # return [e.asdict() for e in f(item)]
                    return pd.DataFrame([e.asdict() for e in f(item, *args, **kwargs)]).to_dict(
                        orient="list"
                    )
                else:
                    return f(item, *args, **kwargs).asdict()

            return decorated

        dataset = super().map(
            function=decorate(function) if as_documents else function,
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

        return Dataset(
            arrow_table=dataset._data,
            info=dataset.info,
            split=dataset.split,
            indices_table=dataset._indices,
            fingerprint=dataset._fingerprint,
        )
