import json
from typing import List

from pytorch_ie.data.document import Document, LabeledSpan

CLASS_NAME_TO_CLASS = {
    "LabeledSpan": LabeledSpan,
}


def load_json(path: str, text_field: str = "text") -> List[Document]:
    with open(path, encoding="utf-8") as data_file:
        data = json.load(data_file)

    annotations = {
        name: CLASS_NAME_TO_CLASS[class_name] for name, class_name in data["annotations"].items()
    }

    documents: List[Document] = []
    for example in data["documents"]:
        text = example.get(text_field)
        assert text is not None, "text_field does not exist."

        document = Document(text)
        for name, cls in annotations.items():
            if name in example:
                field = example[name]
                for dct in field:
                    annotation = cls.from_dict(dct)
                    document.add_annotation(name, annotation)

        documents.append(document)

    return documents
