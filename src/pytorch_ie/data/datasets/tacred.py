import json
import os
from typing import List

from pytorch_ie.data.document import BinaryRelation, Document, LabeledSpan


def _convert_token(token):
    """Convert PTB tokens to normal tokens"""
    return {
        "-lrb-": "(",
        "-rrb-": ")",
        "-lsb-": "[",
        "-rsb-": "]",
        "-lcb-": "{",
        "-rcb-": "}",
    }.get(token.lower(), token)


def load_tacred(path: str, split: str, convert_ptb_tokens: bool = True) -> List[Document]:
    filename = {
        "train": "train.json",
        "validation": "dev.json",
        "test": "test.json",
    }[split]

    file_path = os.path.join(path, filename)

    documents: List[Document] = []
    with open(file_path, encoding="utf-8") as data_file:
        data = json.load(data_file)

        for example in data:
            tokens = example["token"]
            if convert_ptb_tokens:
                tokens = [_convert_token(token) for token in tokens]

            start = 0
            token_offsets = []
            for token in tokens:
                end = start + len(token)
                token_offsets.append((start, end))
                start = end + 1

            text = " ".join(tokens)
            document = Document(text)

            entities: List[LabeledSpan] = []
            for argument in ["subj", "obj"]:
                start = example[argument + "_start"]
                end = example[argument + "_end"]
                label = example[argument + "_type"]

                start_offset = token_offsets[start][0]
                end_offset = token_offsets[end][1]
                entity = LabeledSpan(start=start_offset, end=end_offset, label=label)
                document.add_annotation(
                    "entities",
                    entity,
                )
                entities.append(entity)

            label = example["relation"]
            # if label != "no_relation":
            head, tail = entities
            document.add_annotation("relations", BinaryRelation(head=head, tail=tail, label=label))

            documents.append(document)

    return documents
