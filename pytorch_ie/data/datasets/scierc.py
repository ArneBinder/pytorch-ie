import json
import os
from itertools import chain
from typing import List

from pytorch_ie.data.document import Document, LabeledSpan


def load_scierc(path: str, split: str) -> List[Document]:
    filename = {
        "train": "train.json",
        "validation": "dev.json",
        "test": "test.json",
    }[split]

    file_path = os.path.join(path, filename)

    documents: List[Document] = []
    with open(file_path, encoding="utf-8") as data_file:
        for line in data_file:
            data = json.loads(line)

            tokens = list(chain.from_iterable(data["sentences"]))
            entities = list(chain.from_iterable(data["ner"]))

            start = 0
            token_offsets = []
            for token in tokens:
                end = start + len(token)
                token_offsets.append((start, end))
                start = end + 1

            text = " ".join(tokens)
            document = Document(text)

            start = 0
            for sent_idx, sentence in enumerate(data["sentences"]):
                end = start + len(sentence) - 1
                start_offset = token_offsets[start][0]
                end_offset = token_offsets[end][1]
                document.add_annotation(
                    "sentences",
                    LabeledSpan(start=start_offset, end=end_offset, label=str(sent_idx)),
                )
                start = end + 1

            for start, end, label in entities:
                start_offset = token_offsets[start][0]
                end_offset = token_offsets[end][1]
                document.add_annotation(
                    "entities",
                    LabeledSpan(start=start_offset, end=end_offset, label=label),
                )

            documents.append(document)

    return documents
