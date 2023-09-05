from dataclasses import dataclass

import datasets

import pytorch_ie.data.builder
from pytorch_ie.annotations import LabeledSpan
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.documents import TextDocument
from pytorch_ie.utils.span import tokens_and_tags_to_text_and_labeled_spans


class CoNLLppConfig(datasets.BuilderConfig):
    """BuilderConfig for CoNLLpp"""

    def __init__(self, **kwargs):
        """BuilderConfig for CoNLLpp.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(**kwargs)


@dataclass
class CoNLLppDocument(TextDocument):
    entities: AnnotationList[LabeledSpan] = annotation_field(target="text")


class CoNLLpp(pytorch_ie.data.builder.GeneratorBasedBuilder):
    DOCUMENT_TYPE = CoNLLppDocument

    BASE_DATASET_PATH = "conllpp"

    BUILDER_CONFIGS = [
        CoNLLppConfig(
            name="conllpp", version=datasets.Version("1.0.0"), description="CoNLLpp dataset"
        ),
    ]

    def _generate_document_kwargs(self, dataset):
        return {"int_to_str": dataset.features["ner_tags"].feature.int2str}

    def _generate_document(self, example, int_to_str):
        doc_id = example["id"]
        tokens = example["tokens"]
        ner_tags = [int_to_str(tag) for tag in example["ner_tags"]]

        text, ner_spans = tokens_and_tags_to_text_and_labeled_spans(tokens=tokens, tags=ner_tags)

        document = CoNLLppDocument(text=text, id=doc_id)

        for span in sorted(ner_spans, key=lambda span: span.start):
            document.entities.append(span)

        return document
