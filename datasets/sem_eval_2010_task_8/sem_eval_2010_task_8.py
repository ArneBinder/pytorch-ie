from dataclasses import dataclass

import datasets
import re
import pytorch_ie.data.builder
from pytorch_ie.annotations import LabeledSpan
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.documents import TextDocument
from pytorch_ie.utils.span import tokens_and_tags_to_text_and_labeled_spans

ENTITY_1_PATTERN = re.compile(r"(<e1>(\w+)<\/e1>)")
ENTITY_2_PATTERN = re.compile(r"(<e2>(\w+)<\/e2>)")


@dataclass
class SemEval2010Task8Document(TextDocument):
    entities: AnnotationList[LabeledSpan] = annotation_field(target="text")


class SemEval2010Task8(pytorch_ie.data.builder.GeneratorBasedBuilder):
    DOCUMENT_TYPE = SemEval2010Task8Document

    BASE_DATASET_PATH = "conll2003"

    VERSION = datasets.Version("1.0.0")

    def _generate_document_kwargs(self, dataset):
        return {"int_to_str": dataset.features["relation"].feature.int2str}

    def _generate_document(self, example, int_to_str):
        text = example["tokens"]
        relation = int_to_str(example["relation"])

        document = SemEval2010Task8Document(text=text)

        # for span in sorted(ner_spans, key=lambda span: span.start):
        #     document.entities.append(span)

        return document

    @staticmethod
    def _get_entity_spans_from_text(text):
        text = (text.replace("<e1>", "<e1> ")
                    .replace("<e2>", "<e2> ")
                    .replace("</e1>", " </e1>")
                    .replace("</e2>", " </e2>")
        )

        match_entity_1 = ENTITY_1_PATTERN.match(text)
        match_entity_2 = ENTITY_2_PATTERN.match(text)
