from typing import List, Union

from datasets import load_dataset
from datasets.splits import Split

from pytorch_ie.data.document import Document, Label


def load_sst2(
    split: Union[str, Split],
) -> List[Document]:
    data = load_dataset("glue", "sst2", split=split)

    label_field = "label"

    int_to_str = data.features[label_field].int2str

    documents = []
    for example in data:
        text = example["sentence"]
        document = Document(text)

        label = int_to_str(example[label_field])

        document.add_annotation("labels", Label(label=label))

        documents.append(document)

    return documents
