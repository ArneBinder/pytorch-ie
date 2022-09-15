# Lint as: python3
import json
import logging
import os

from detectron2.data.detection_utils import read_image
from detectron2.data.transforms import ResizeTransform, TransformList
from torch import tensor

import datasets


def normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]


def simplify_bbox(bbox):
    return [
        min(bbox[0::2]),
        min(bbox[1::2]),
        max(bbox[2::2]),
        max(bbox[3::2]),
    ]


def merge_bbox(bbox_list):
    x0, y0, x1, y1 = list(zip(*bbox_list))
    return [min(x0), min(y0), max(x1), max(y1)]


def load_image(image_path):
    image = read_image(image_path, format="BGR")
    h = image.shape[0]
    w = image.shape[1]
    img_trans = TransformList([ResizeTransform(h=h, w=w, new_h=224, new_w=224)])
    image = tensor(img_trans.apply_image(image).copy()).permute(
        2, 0, 1
    )  # copy to make it writeable
    return image, (w, h)


_URL = "https://github.com/doc-analysis/XFUN/releases/download/v1.0/"

_LANG = ["zh", "de", "es", "fr", "en", "it", "ja", "pt"]
logger = logging.getLogger(__name__)


class XFUNConfig(datasets.BuilderConfig):
    """BuilderConfig for XFUN."""

    def __init__(self, lang, additional_langs=None, **kwargs):
        """
        Args:
            lang: string, language for the input text
            **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(**kwargs)
        self.lang = lang
        self.additional_langs = additional_langs


_LABELS = ["header", "question", "answer", "other"]


def _get_box_feature():
    return datasets.Sequence(datasets.Value("int64"))


class XFUN(datasets.GeneratorBasedBuilder):
    """XFUN dataset."""

    BUILDER_CONFIGS = [XFUNConfig(name=f"xfun.{lang}", lang=lang) for lang in _LANG]

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "uid": datasets.Value("string"),
                    "document": datasets.Sequence(
                        {
                            "id": datasets.Value("int64"),
                            "box": _get_box_feature(),
                            "text": datasets.Value("string"),
                            "label": datasets.ClassLabel(names=_LABELS),
                            "words": datasets.Sequence(
                                {
                                    "box": _get_box_feature(),
                                    "text": datasets.Value("string"),
                                }
                            ),
                            "linking": datasets.Sequence(
                                datasets.Sequence(datasets.Value("int64"))
                            ),
                        }
                    ),
                    "img_meta": {
                        "fname": datasets.Value("string"),
                        "width": datasets.Value("int64"),
                        "height": datasets.Value("int64"),
                    },
                    # has to be at the root level, crashes otherwise
                    "img_data": datasets.Array3D(shape=(3, 224, 224), dtype="uint8"),
                },
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            "train": [
                f"{_URL}{self.config.lang}.train.json",
                f"{_URL}{self.config.lang}.train.zip",
            ],
            "val": [f"{_URL}{self.config.lang}.val.json", f"{_URL}{self.config.lang}.val.zip"],
            # "test": [f"{_URL}{self.config.lang}.test.json", f"{_URL}{self.config.lang}.test.zip"],
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)
        train_files_for_many_langs = [downloaded_files["train"]]
        val_files_for_many_langs = [downloaded_files["val"]]
        # test_files_for_many_langs = [downloaded_files["test"]]
        if self.config.additional_langs:
            additional_langs = self.config.additional_langs.split("+")
            if "all" in additional_langs:
                additional_langs = [lang for lang in _LANG if lang != self.config.lang]
            for lang in additional_langs:
                urls_to_download = {
                    "train": [f"{_URL}{lang}.train.json", f"{_URL}{lang}.train.zip"]
                }
                additional_downloaded_files = dl_manager.download_and_extract(urls_to_download)
                train_files_for_many_langs.append(additional_downloaded_files["train"])

        logger.info(
            f"Training on {self.config.lang} with additional langs({self.config.additional_langs})"
        )
        logger.info(f"Evaluating on {self.config.lang}")
        logger.info(f"Testing on {self.config.lang}")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepaths": train_files_for_many_langs}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"filepaths": val_files_for_many_langs}
            ),
            # datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepaths": test_files_for_many_langs}),
        ]

    def _generate_examples(self, filepaths):
        for filepath in filepaths:
            logger.info("Generating examples from = %s", filepath)
            with open(filepath[0], encoding="utf-8") as f:
                data = json.load(f)

            for doc in data["documents"]:
                # print(json.dumps(doc, indent=2))
                fpath = os.path.join(filepath[1], doc["img"]["fname"])
                image, size = load_image(fpath)
                expected_size = tuple([doc["img"]["width"], doc["img"]["height"]])
                if size != expected_size:
                    raise ValueError(
                        f"image has unexpected size: {size}. expected: {expected_size}"
                    )

                doc["img_meta"] = doc.pop("img")
                doc["img_data"] = image

                yield doc["id"], doc
