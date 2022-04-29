from dataclasses import dataclass

import datasets
import pytorch_ie.data.builder
from pytorch_ie import AnnotationList, LabeledSpan, TextDocument, annotation_field
from pytorch_ie.utils.span import bio_tags_to_spans


class Conll2003Config(datasets.BuilderConfig):
    """BuilderConfig for Conll2003"""

    def __init__(self, **kwargs):
        """BuilderConfig forConll2003.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(**kwargs)


@dataclass
class CoNLL2003Document(TextDocument):
    entities: AnnotationList[LabeledSpan] = annotation_field(target="text")


class Conll2003(pytorch_ie.data.builder.GeneratorBasedBuilder):
    DOCUMENT_TYPE = CoNLL2003Document

    BASE_DATASET_PATH = "conll2003"

    BUILDER_CONFIGS = [
        Conll2003Config(
            name="conll2003", version=datasets.Version("1.0.0"), description="Conll2003 dataset"
        ),
    ]

    def _generate_document_kwargs(self, dataset):
        return {"int_to_str": dataset.features["ner_tags"].feature.int2str}

    def _generate_document(self, example, int_to_str):
        doc_id = example["id"]
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

        document = CoNLL2003Document(text=text, id=doc_id)

        for label, (start, end) in spans:
            start_offset = token_offsets[start][0]
            end_offset = token_offsets[end][1]
            document.entities.append(LabeledSpan(start=start_offset, end=end_offset, label=label))

        return document
