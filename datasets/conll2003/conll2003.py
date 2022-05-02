from dataclasses import dataclass

import datasets
import pytorch_ie.data.builder
from pytorch_ie import AnnotationList, LabeledSpan, TextDocument, annotation_field
from pytorch_ie.utils.span import tokens_and_tags_to_text_and_labeled_spans


class CoNLL2003Config(datasets.BuilderConfig):
    """BuilderConfig for CoNLL2003"""

    def __init__(self, **kwargs):
        """BuilderConfig for CoNLL2003.
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
        CoNLL2003Config(
            name="conll2003", version=datasets.Version("1.0.0"), description="CoNLL2003 dataset"
        ),
    ]

    def _generate_document_kwargs(self, dataset):
        return {"int_to_str": dataset.features["ner_tags"].feature.int2str}

    def _generate_document(self, example, int_to_str):
        doc_id = example["id"]
        tokens = example["tokens"]
        ner_tags = [int_to_str(tag) for tag in example["ner_tags"]]

        text, ner_spans = tokens_and_tags_to_text_and_labeled_spans(tokens=tokens, tags=ner_tags)

        document = CoNLL2003Document(text=text, id=doc_id)

        for span in sorted(ner_spans, key=lambda span: span.start):
            document.entities.append(span)

        return document
