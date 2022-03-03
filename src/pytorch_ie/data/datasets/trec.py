from typing import List, Union

from datasets import Dataset, IterableDataset, load_dataset
from datasets.splits import Split

from pytorch_ie.data.document import Document, Label


def load_trec(
    split: Union[str, Split],
    fine_grained: bool = False,
) -> List[Document]:
    path = "trec"
    data = load_dataset(path, split=split)

    field_coarse = "label-coarse"
    field_fine = "label-fine"

    assert isinstance(data, (Dataset, IterableDataset)), (
        f"`load_dataset` for `path={path}` and `split={split}` should return a single Dataset, "
        f"but the result is of type: {type(data)}"
    )

    int_to_str_coarse = data.features[field_coarse].int2str
    int_to_str_fine = data.features[field_fine].int2str

    documents = []
    for example in data:
        text = example["text"]
        document = Document(text)

        label = int_to_str_coarse(example[field_coarse])

        if fine_grained:
            label_fine = int_to_str_fine(example[field_fine])
            label = f"{label}:{label_fine}"

        document.add_annotation("labels", Label(label=label))

        documents.append(document)

    return documents
