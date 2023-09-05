from dataclasses import dataclass

import datasets

import pytorch_ie.data.builder
from pytorch_ie.annotations import LabeledSpan
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.documents import TextDocument
from pytorch_ie.utils.span import tokens_and_tags_to_text_and_labeled_spans


class GermEval14Config(datasets.BuilderConfig):
    """BuilderConfig for GermEval 2014."""

    def __init__(self, **kwargs):
        """BuilderConfig for GermEval 2014.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(**kwargs)


@dataclass
class GermEval14Document(TextDocument):
    entities: AnnotationList[LabeledSpan] = annotation_field(target="text")


class GermEval14(pytorch_ie.data.builder.GeneratorBasedBuilder):
    DOCUMENT_TYPE = GermEval14Document

    BASE_DATASET_PATH = "germeval_14"

    BUILDER_CONFIGS = [
        GermEval14Config(
            name="germeval_14",
            version=datasets.Version("2.0.0"),
            description="GermEval 2014 NER Shared Task dataset",
        ),
    ]

    def _generate_document_kwargs(self, dataset):
        return {"int_to_str": dataset.features["ner_tags"].feature.int2str}

    def _generate_document(self, example, int_to_str):
        doc_id = example["id"]
        tokens = example["tokens"]
        ner_tags = [int_to_str(tag) for tag in example["ner_tags"]]
        nested_ner_tags = [int_to_str(tag) for tag in example["nested_ner_tags"]]

        text, ner_spans = tokens_and_tags_to_text_and_labeled_spans(tokens=tokens, tags=ner_tags)
        _, nested_ner_tags = tokens_and_tags_to_text_and_labeled_spans(
            tokens=tokens, tags=nested_ner_tags
        )

        document = GermEval14Document(text=text, id=doc_id)

        for span in sorted(ner_spans + nested_ner_tags, key=lambda span: span.start):
            document.entities.append(span)

        return document
