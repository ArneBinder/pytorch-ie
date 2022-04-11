from typing import Any, Callable, Dict, List, Optional, Union

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


def load_hf_conll2003(
    split: Optional[Union[str, Split]] = None,
) -> Union[DatasetDict, IterableDatasetDict]:
    data = single_split_to_dict(load_dataset("conll2003", split=split), split=split)
    return data


def _hf_example_to_document(
    example: Dict[str, Any], int_to_str: Callable[[int], str], entity_layer: str = "entities"
) -> Document:
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
    document.annotations.spans.create_layer(name=entity_layer)

    for label, (start, end) in spans:
        start_offset = token_offsets[start][0]
        end_offset = token_offsets[end][1]
        document.add_annotation(
            entity_layer, LabeledSpan(start=start_offset, end=end_offset, label=label)
        )
    return document


def _hf_to_document_dataset(
    hf_dataset: Union[Dataset, IterableDataset],
    entity_layer: str = "entities",
) -> List[Document]:
    int_to_str = hf_dataset.features["ner_tags"].feature.int2str
    documents = []
    for example in hf_dataset:
        document = _hf_example_to_document(
            example=example, int_to_str=int_to_str, entity_layer=entity_layer
        )
        documents.append(document)
    return documents


def from_hf(
    hf_dataset_dict: Union[DatasetDict, IterableDatasetDict],
    entity_layer: str = "entities",
) -> PIEDatasetDict:
    return {
        k: _hf_to_document_dataset(v, entity_layer=entity_layer)
        for k, v in hf_dataset_dict.items()
    }


# TODO: this should return a PIEDatasetDict
def load_conll2003(
    split: Union[str, Split],
) -> List[Document]:
    dataset_hf = load_hf_conll2003(split=split)
    dataset = from_hf(hf_dataset_dict=dataset_hf)

    return dataset[split]
