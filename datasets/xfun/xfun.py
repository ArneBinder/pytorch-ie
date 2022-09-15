from dataclasses import dataclass
from typing import Optional, Tuple

import pytorch_ie.data.builder
from pytorch_ie.core import AnnotationList, Document, annotation_field

import datasets
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

from pytorch_ie.core import Annotation
#from pytorch_ie.core.document import resolve_annotation


def resolve_annotation(
    id_or_annotation: Union[int, Annotation],
    store: Optional[Dict[int, Annotation]],
) -> Annotation:
    if isinstance(id_or_annotation, Annotation):
        return id_or_annotation
    else:
        if store is None:
            raise ValueError("Unable to resolve the annotation id without annotation_store.")
        return store[id_or_annotation][1]


def _post_init_bbox(self):
    if not isinstance(self.bbox, tuple):
        object.__setattr__(self, "bbox", tuple(self.bbox))
    if not len(self.bbox) == 4:
        raise ValueError("bounding box has to consist of 4 values.")


@dataclass(eq=True, frozen=True)
class OcrWord(Annotation):
    bbox: Tuple[int, int, int, int]
    text: str

    def __post_init__(self) -> None:
        _post_init_bbox(self)


@dataclass(eq=True, frozen=True)
class OcrPhrase(Annotation):
    words: Tuple[OcrWord, ...]
    bbox: Tuple[int, int, int, int]
    label: Optional[str] = None
    text: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.words, tuple):
            object.__setattr__(self, "words", tuple(self.words))
        _post_init_bbox(self)

    def asdict(self) -> Dict[str, Any]:
        dct = super().asdict()
        dct["words"] = tuple(hash(word) for word in self.words)
        return dct

    @classmethod
    def fromdict(
        cls,
        dct: Dict[str, Any],
        annotation_store: Optional[Dict[int, "Annotation"]] = None,
    ):
        tmp_dct = dict(dct)
        tmp_dct.pop("_id", None)

        tmp_dct["words"] = [
            resolve_annotation(word, store=annotation_store) for word in tmp_dct["words"]
        ]

        return cls(**tmp_dct)


@dataclass(eq=True, frozen=True)
class OcrBinaryRelation(Annotation):
    head: OcrPhrase
    tail: OcrPhrase
    label: Optional[str] = None
    score: float = 1.0

    def asdict(self) -> Dict[str, Any]:
        dct = super().asdict()
        dct["head"] = hash(self.head)
        dct["tail"] = hash(self.tail)
        return dct

    @classmethod
    def fromdict(
        cls,
        dct: Dict[str, Any],
        annotation_store: Optional[Dict[int, "Annotation"]] = None,
    ):
        tmp_dct = dict(dct)
        tmp_dct.pop("_id", None)

        tmp_dct["head"] = resolve_annotation(tmp_dct["head"], store=annotation_store)
        tmp_dct["tail"] = resolve_annotation(tmp_dct["tail"], store=annotation_store)

        return cls(**tmp_dct)


_LANG = ["zh", "de", "es", "fr", "en", "it", "ja", "pt"]


@dataclass
class XFUNDocument(Document):
    id: str
    # 3d: channel x row x col
    image: Tuple[
        Tuple[Tuple[int, ...], ...], Tuple[Tuple[int, ...], ...], Tuple[Tuple[int, ...], ...]
    ]
    image_width: int
    image_height: int
    words: AnnotationList[OcrWord] = annotation_field(target="image")
    phrases: AnnotationList[OcrPhrase] = annotation_field(target="words")
    relations: AnnotationList[OcrBinaryRelation] = annotation_field(target="phrases")
    image_fname: Optional[str] = None

    def __post_init__(self):
        # when creating from a dataset, this comes in as a list (json does not know tuples)
        if not isinstance(self.image, tuple):
            object.__setattr__(
                self,
                "image",
                tuple(tuple(tuple(row) for row in channel) for channel in self.image),
            )
        super().__post_init__()


class XFUNConfig(datasets.BuilderConfig):
    """BuilderConfig for XFUN"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


def dl2ld(dl):
    # convert a dict of lists to a list of dicts
    return [dict(zip(dl, t)) for t in zip(*dl.values())]


class XFUN(pytorch_ie.data.builder.GeneratorBasedBuilder):
    DOCUMENT_TYPE = XFUNDocument

    BASE_DATASET_PATH = "hf_datasets/xfun"

    BUILDER_CONFIGS = [
        XFUNConfig(name=f"xfun.{lang}", version=datasets.Version("1.0.0")) for lang in _LANG
    ]

    def __init__(self, **kwargs):
        builder_kwargs = dict(kwargs)
        builder_kwargs.pop("hash", None)
        builder_kwargs.pop("base_path", None)
        config_name = builder_kwargs.pop("config_name", None)
        print(builder_kwargs)
        self.base_builder = None
        if self.BASE_DATASET_PATH is not None:
            self.base_builder = datasets.load.load_dataset_builder(
                path=self.BASE_DATASET_PATH,
                name=config_name,
                **builder_kwargs,
            )

        super().__init__(**kwargs)

    def _generate_document_kwargs(self, dataset):
        return {"int_to_str": dataset.features["document"].feature["label"].int2str}

    def _generate_document(self, example, int_to_str):
        document = XFUNDocument(
            id=example["id"],
            image=example["img_data"],
            image_width=example["img_meta"]["width"],
            image_height=example["img_meta"]["height"],
            image_fname=example["img_meta"]["fname"],
        )
        all_words = []
        phrases = {}
        relation_id_pairs = []
        for _phrase in dl2ld(example["document"]):
            # skip empty phrases
            if len(_phrase["text"]) == 0:
                continue
            words = []
            for _word in dl2ld(_phrase["words"]):
                word = OcrWord(bbox=_word["box"], text=_word["text"])
                words.append(word)
            all_words.extend(words)
            label = int_to_str(_phrase["label"])
            phrase = OcrPhrase(
                words=tuple(words), bbox=_phrase["box"], label=label, text=_phrase["text"]
            )
            phrases[_phrase["id"]] = phrase
            relation_id_pairs.extend(_phrase["linking"])

        valid_arg_pairs = set()
        for arg1_id, arg2_id in relation_id_pairs:
            if arg1_id not in phrases or arg2_id not in phrases:
                continue
            label_pair = [phrases[arg1_id].label, phrases[arg2_id].label]
            if label_pair == ["question", "answer"]:
                valid_arg_pairs.add((arg1_id, arg2_id))
            elif label_pair == ["answer", "question"]:
                valid_arg_pairs.add((arg2_id, arg1_id))
            else:
                continue

        relations = [
            OcrBinaryRelation(head=phrases[head_id], tail=phrases[tail_id])
            for head_id, tail_id in valid_arg_pairs
        ]

        document.words.extend(all_words)
        document.phrases.extend(phrases.values())
        document.relations.extend(relations)

        return document
