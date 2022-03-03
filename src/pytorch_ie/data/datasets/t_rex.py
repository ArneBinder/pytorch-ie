import json
from typing import List

from pytorch_ie.data.document import Document, LabeledSpan

CLASS_NAME_TO_CLASS = {
    "LabeledSpan": LabeledSpan,
}


def load_t_rex(path: str, text_field: str = "text") -> List[Document]:
    with open(path, encoding="utf-8") as data_file:
        data = json.load(data_file)

    documents: List[Document] = []
    for example in data:
        text = example.get(text_field)
        assert text is not None, "text_field does not exist."

        document = Document(text)
        for entity in example["entities"]:
            start, end = entity["boundaries"]
            document.add_annotation("entities", LabeledSpan(start, end, label="ENT"))

        for sent_idx, (start, end) in enumerate(example["sentences_boundaries"]):
            document.add_annotation("sentences", LabeledSpan(start, end, label=str(sent_idx)))

        documents.append(document)

    return documents
