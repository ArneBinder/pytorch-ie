from dataclasses import dataclass

import datasets
import pytorch_ie.data.builder
from pytorch_ie.annotations import LabeledSpan
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.documents import TextDocument
from pytorch_ie.utils.span import tokens_and_tags_to_text_and_labeled_spans

_VERSION = "1.0.0"
_COURTS = ["bag", "bfh", "bgh", "bpatg", "bsg", "bverfg", "bverwg"]
_COURTS_FILEPATHS = {court: f"{court}.conll" for court in _COURTS}
_ALL = "all"


class GermanLegalEntityRecognitionConfig(datasets.BuilderConfig):
    def __init__(self, *args, courts=None, **kwargs):
        super().__init__(*args, version=datasets.Version(_VERSION, ""), **kwargs)
        self.courts = courts

    @property
    def filepaths(self):
        return [_COURTS_FILEPATHS[court] for court in self.courts]


@dataclass
class GermanLegalEntityRecognitionDocument(TextDocument):
    entities: AnnotationList[LabeledSpan] = annotation_field(target="text")


class GermanLegalEntityRecognition(pytorch_ie.data.builder.GeneratorBasedBuilder):
    DOCUMENT_TYPE = GermanLegalEntityRecognitionDocument

    BASE_DATASET_PATH = "german_legal_entity_recognition"

    BUILDER_CONFIGS = [
        GermanLegalEntityRecognitionConfig(
            name=court, courts=[court], description=f"Court. {court}."
        )
        for court in _COURTS
    ] + [
        GermanLegalEntityRecognitionConfig(
            name=_ALL, courts=_COURTS, description="All courts included."
        )
    ]
    BUILDER_CONFIG_CLASS = GermanLegalEntityRecognitionConfig
    DEFAULT_CONFIG_NAME = _ALL  # type: ignore

    def _generate_document_kwargs(self, dataset):
        return {"int_to_str": dataset.features["ner_tags"].feature.int2str}

    def _generate_document(self, example, int_to_str):
        doc_id = example["id"]
        tokens = example["tokens"]
        ner_tags = [int_to_str(tag) for tag in example["ner_tags"]]

        text, ner_spans = tokens_and_tags_to_text_and_labeled_spans(tokens=tokens, tags=ner_tags)

        document = GermanLegalEntityRecognitionDocument(text=text, id=doc_id)

        for span in sorted(ner_spans, key=lambda span: span.start):
            document.entities.append(span)

        return document
