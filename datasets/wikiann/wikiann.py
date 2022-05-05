from dataclasses import dataclass

import datasets
import pytorch_ie.data.builder
from pytorch_ie.annotations import LabeledSpan
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.documents import TextDocument
from pytorch_ie.utils.span import tokens_and_tags_to_text_and_labeled_spans

_VERSION = "1.1.0"
_LANGS = [
    "ace",
    "af",
    "als",
    "am",
    "an",
    "ang",
    "ar",
    "arc",
    "arz",
    "as",
    "ast",
    "ay",
    "az",
    "ba",
    "bar",
    "bat-smg",
    "be",
    "be-x-old",
    "bg",
    "bh",
    "bn",
    "bo",
    "br",
    "bs",
    "ca",
    "cbk-zam",
    "cdo",
    "ce",
    "ceb",
    "ckb",
    "co",
    "crh",
    "cs",
    "csb",
    "cv",
    "cy",
    "da",
    "de",
    "diq",
    "dv",
    "el",
    "eml",
    "en",
    "eo",
    "es",
    "et",
    "eu",
    "ext",
    "fa",
    "fi",
    "fiu-vro",
    "fo",
    "fr",
    "frr",
    "fur",
    "fy",
    "ga",
    "gan",
    "gd",
    "gl",
    "gn",
    "gu",
    "hak",
    "he",
    "hi",
    "hr",
    "hsb",
    "hu",
    "hy",
    "ia",
    "id",
    "ig",
    "ilo",
    "io",
    "is",
    "it",
    "ja",
    "jbo",
    "jv",
    "ka",
    "kk",
    "km",
    "kn",
    "ko",
    "ksh",
    "ku",
    "ky",
    "la",
    "lb",
    "li",
    "lij",
    "lmo",
    "ln",
    "lt",
    "lv",
    "map-bms",
    "mg",
    "mhr",
    "mi",
    "min",
    "mk",
    "ml",
    "mn",
    "mr",
    "ms",
    "mt",
    "mwl",
    "my",
    "mzn",
    "nap",
    "nds",
    "ne",
    "nl",
    "nn",
    "no",
    "nov",
    "oc",
    "or",
    "os",
    "pa",
    "pdc",
    "pl",
    "pms",
    "pnb",
    "ps",
    "pt",
    "qu",
    "rm",
    "ro",
    "ru",
    "rw",
    "sa",
    "sah",
    "scn",
    "sco",
    "sd",
    "sh",
    "si",
    "simple",
    "sk",
    "sl",
    "so",
    "sq",
    "sr",
    "su",
    "sv",
    "sw",
    "szl",
    "ta",
    "te",
    "tg",
    "th",
    "tk",
    "tl",
    "tr",
    "tt",
    "ug",
    "uk",
    "ur",
    "uz",
    "vec",
    "vep",
    "vi",
    "vls",
    "vo",
    "wa",
    "war",
    "wuu",
    "xmf",
    "yi",
    "yo",
    "zea",
    "zh",
    "zh-classical",
    "zh-min-nan",
    "zh-yue",
]


class WikiANNConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version(_VERSION, ""), **kwargs)


@dataclass
class WikiANNDocument(TextDocument):
    entities: AnnotationList[LabeledSpan] = annotation_field(target="text")


class WikiANN(pytorch_ie.data.builder.GeneratorBasedBuilder):
    DOCUMENT_TYPE = WikiANNDocument

    BASE_DATASET_PATH = "wikiann"

    BUILDER_CONFIGS = [
        WikiANNConfig(name=lang, description=f"WikiANN NER examples in language {lang}")
        for lang in _LANGS
    ]

    def _generate_document_kwargs(self, dataset):
        return {"int_to_str": dataset.features["ner_tags"].feature.int2str}

    def _generate_document(self, example, int_to_str):
        tokens = example["tokens"]
        ner_tags = [int_to_str(tag) for tag in example["ner_tags"]]

        text, ner_spans = tokens_and_tags_to_text_and_labeled_spans(tokens=tokens, tags=ner_tags)

        document = WikiANNDocument(text=text, id=None)

        for span in sorted(ner_spans, key=lambda span: span.start):
            document.entities.append(span)

        return document
