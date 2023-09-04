from dataclasses import dataclass

import datasets

import pytorch_ie.data.builder
from pytorch_ie.annotations import LabeledSpan
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.documents import TextDocument
from pytorch_ie.utils.span import tokens_and_tags_to_text_and_labeled_spans


class Conll2002Config(datasets.BuilderConfig):
    """BuilderConfig for CoNLL2002"""

    def __init__(self, **kwargs):
        """BuilderConfig for CoNLL2002.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(**kwargs)


@dataclass
class CoNLL2002Document(TextDocument):
    entities: AnnotationList[LabeledSpan] = annotation_field(target="text")


class Conll2003(pytorch_ie.data.builder.GeneratorBasedBuilder):
    DOCUMENT_TYPE = CoNLL2002Document

    BASE_DATASET_PATH = "conll2002"

    BUILDER_CONFIGS = [
        Conll2002Config(
            name="es", version=datasets.Version("1.0.0"), description="CoNLL2002 Spanish dataset"
        ),
        Conll2002Config(
            name="nl", version=datasets.Version("1.0.0"), description="CoNLL2002 Dutch dataset"
        ),
    ]

    def _generate_document_kwargs(self, dataset):
        return {"int_to_str": dataset.features["ner_tags"].feature.int2str}

    def _generate_document(self, example, int_to_str):
        doc_id = example["id"]
        tokens = example["tokens"]
        ner_tags = [int_to_str(tag) for tag in example["ner_tags"]]

        text, ner_spans = tokens_and_tags_to_text_and_labeled_spans(tokens=tokens, tags=ner_tags)

        document = CoNLL2002Document(text=text, id=doc_id)

        for span in sorted(ner_spans, key=lambda span: span.start):
            document.entities.append(span)

        return document
