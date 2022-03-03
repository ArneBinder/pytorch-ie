from typing import List, Union

from datasets import Dataset, IterableDataset, load_dataset
from datasets.splits import Split

from pytorch_ie.data.document import Document, LabeledSpan
from pytorch_ie.utils.span import io_tags_to_spans


def load_few_nerd(
    split: Union[str, Split],
    name: str,
    fine_grained: bool = False,
) -> List[Document]:
    path = "dfki-nlp/few-nerd"
    data = load_dataset(path, name=name, split=split)

    ner_field = "ner_tags"
    if fine_grained:
        ner_field = "ner_fine"

    assert isinstance(data, (Dataset, IterableDataset)), (
        f"`load_dataset` for `path={path}` and `split={split}` should return a single Dataset, "
        f"but the result is of type: {type(data)}"
    )
    int_to_str = data.features[ner_field].feature.int2str

    documents = []
    for example in data:
        tokens = example["tokens"]
        ner_tags = example[ner_field]

        start = 0
        token_offsets = []
        tag_sequence = []
        for token, tag_id in zip(tokens, ner_tags):
            end = start + len(token)
            token_offsets.append((start, end))
            tag_sequence.append(int_to_str(tag_id))

            start = end + 1

        text = " ".join(tokens)

        spans = io_tags_to_spans(tag_sequence)

        document = Document(text)

        for label, (start, end) in spans:
            start_offset = token_offsets[start][0]
            end_offset = token_offsets[end][1]
            document.add_annotation(
                "entities", LabeledSpan(start=start_offset, end=end_offset, label=label)
            )

        documents.append(document)

    return documents
