from typing import List, Union

from datasets import load_dataset
from datasets.splits import Split

from pytorch_ie.data.document import Document, Label


def load_sst(
    split: Union[str, Split],
) -> List[Document]:
    data = load_dataset("sst", split=split)

    documents = []
    for example in data:
        text = example["sentence"]
        document = Document(text)

        label = "positive" if example["label"] > 0.5 else "negative"
        document.add_annotation("labels", Label(label=label))

        documents.append(document)

    return documents
