from dataclasses import dataclass

import datasets
import pytorch_ie.data.builder
from pytorch_ie.annotations import Label
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.documents import TextDocument


class ImdbConfig(datasets.BuilderConfig):
    """BuilderConfig for IMDB"""

    def __init__(self, **kwargs):
        """BuilderConfig for IMDB.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(**kwargs)


@dataclass
class ImdbDocument(TextDocument):
    label: AnnotationList[Label] = annotation_field()


class Imdb(pytorch_ie.data.builder.GeneratorBasedBuilder):
    DOCUMENT_TYPE = ImdbDocument

    BASE_DATASET_PATH = "imdb"

    BUILDER_CONFIGS = [
        ImdbConfig(
            name="plain_text",
            version=datasets.Version("1.0.0"),
            description="IMDB sentiment classification dataset",
        ),
    ]

    def _generate_document_kwargs(self, dataset):
        return {"int2str": dataset.features["label"].int2str}

    def _generate_document(self, example, int2str):
        text = example["text"]
        document = ImdbDocument(text=text)
        label_id = example["label"]
        if label_id < 0:
            return document

        label = int2str(label_id)
        label_annotation = Label(label=label)
        document.label.append(label_annotation)

        return document
