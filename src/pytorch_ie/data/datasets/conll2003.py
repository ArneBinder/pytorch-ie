from typing import List, Optional, Union

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset
from datasets.splits import Split

from pytorch_ie.data.datasets import PIEDatasetDict
from pytorch_ie.data.document import Document, LabeledSpan
from pytorch_ie.utils.span import bio_tags_to_spans


def single_split_to_dict(
    dataset: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset],
    split: Optional[Union[str, Split]] = None,
) -> Union[DatasetDict, IterableDatasetDict]:
    if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
        return dataset
    if split is None:
        raise ValueError(
            "split name has to be provided to convert an (Iterable)Dataset to an (Iterable)DatasetDict"
        )
    if isinstance(dataset, Dataset):
        return DatasetDict({split: dataset})
    if isinstance(dataset, IterableDataset):
        return IterableDatasetDict({split: dataset})
    raise ValueError(f"dataset has unknown type: {type(dataset)}")


def load_conll2003_hf(
    split: Optional[Union[str, Split]] = None,
) -> Union[DatasetDict, IterableDatasetDict]:
    data = single_split_to_dict(load_dataset("conll2003", split=split), split=split)
    return data


def _convert_conll2003_hf_to_document_dataset(
    dataset_hf: Union[Dataset, IterableDataset]
) -> List[Document]:
    int_to_str = dataset_hf.features["ner_tags"].feature.int2str

    documents = []
    for example in dataset_hf:
        tokens = example["tokens"]
        ner_tags = example["ner_tags"]

        start = 0
        token_offsets = []
        tag_sequence = []
        for token, tag_id in zip(tokens, ner_tags):
            end = start + len(token)
            token_offsets.append((start, end))
            tag_sequence.append(int_to_str(tag_id))

            start = end + 1

        text = " ".join(tokens)
        spans = bio_tags_to_spans(tag_sequence)

        document = Document(text)

        for label, (start, end) in spans:
            start_offset = token_offsets[start][0]
            end_offset = token_offsets[end][1]
            document.add_annotation(
                "entities", LabeledSpan(start=start_offset, end=end_offset, label=label)
            )

        documents.append(document)
    return documents


def convert_conll2003_hf_to_document_dataset(
    dataset_hf: Union[DatasetDict, IterableDatasetDict]
) -> PIEDatasetDict:
    return {k: _convert_conll2003_hf_to_document_dataset(v) for k, v in dataset_hf.items()}


# TODO: this should return a PIEDatasetDict
def load_conll2003(
    split: Union[str, Split],
) -> List[Document]:
    dataset_hf = load_conll2003_hf(split=split)
    dataset = convert_conll2003_hf_to_document_dataset(dataset_hf=dataset_hf)

    return dataset[split]
