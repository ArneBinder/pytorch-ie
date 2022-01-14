from typing import List, Union

from datasets import Dataset, IterableDataset, load_dataset
from datasets.splits import Split

from pytorch_ie.data.document import Document, Label


def load_ag_news(
    split: Union[str, Split],
) -> List[Document]:
    path = "ag_news"
    data = load_dataset(path, split=split)
    assert isinstance(data, (Dataset, IterableDataset)), (
        f"`load_dataset` for `path={path}` and `split={split}` should return a single Dataset, "
        f"but the result is of type: {type(data)}"
    )

    label_field = "label"

    int_to_str = data.features[label_field].int2str

    documents = []
    for example in data:
        text = example["text"]
        document = Document(text)

        label = int_to_str(example[label_field])

        document.add_annotation("labels", Label(label=label))

        documents.append(document)

    return documents
