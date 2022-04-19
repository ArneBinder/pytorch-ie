import datasets
from datasets.load import load_dataset_builder
import abc
from typing import Optional, Mapping
from functools import wraps
from pytorch_ie.data.dataset import Dataset


class GeneratorBasedBuilder(datasets.builder.GeneratorBasedBuilder):
    # Default batch size used by the ArrowWriter
    # It defines the number of samples that are kept in memory before writing them
    # and also the length of the arrow chunks
    # None means that the ArrowWriter will use its default value
    DOCUMENT_TYPE = None

    BASE_PATH = None

    def __init__(self, **kwargs):
        builder_kwargs = dict(kwargs)
        builder_kwargs.pop("hash", None)
        builder_kwargs.pop("base_path", None)
        self.base_builder = load_dataset_builder(
            path=self.BASE_PATH,
            **builder_kwargs,
        )
        super().__init__(**kwargs)

    def _info(self):
        return self.base_builder._info()

    def _split_generators(self, dl_manager):
        return self.base_builder._split_generators(dl_manager)

    def _generate_examples(self, filepath):
        return self.base_builder._generate_examples(filepath)

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

        mapped_dataset = dataset.map(decorate(self._generate_document), fn_kwargs=fn_kwargs)

        document_dataset = Dataset.from_hf_dataset(mapped_dataset)
        document_dataset.set_format("document", document_type=self.DOCUMENT_TYPE)

        return document_dataset
