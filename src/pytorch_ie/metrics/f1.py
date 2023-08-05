import json
import logging
from collections import defaultdict
from functools import partial
from typing import Callable, Collection, Dict, Hashable, Optional, Set, Tuple

import pandas as pd

from pytorch_ie.core import Annotation, Document, DocumentMetric

logger = logging.getLogger(__name__)


def eval_counts_for_layer(
    document: Document,
    layer: str,
    annotation_filter: Optional[Callable[[Annotation], bool]] = None,
    annotation_mapper: Optional[Callable[[Annotation], Hashable]] = None,
) -> Tuple[int, int, int]:
    annotation_filter = annotation_filter or (lambda ann: True)
    annotation_mapper = annotation_mapper or (lambda ann: ann)
    predicted_annotations = {
        annotation_mapper(ann) for ann in document[layer].predictions if annotation_filter(ann)
    }
    gold_annotations = {
        annotation_mapper(ann) for ann in document[layer] if annotation_filter(ann)
    }
    tp = len([ann for ann in predicted_annotations & gold_annotations])
    fn = len([ann for ann in gold_annotations - predicted_annotations])
    fp = len([ann for ann in predicted_annotations - gold_annotations])
    return tp, fp, fn


def _remove_annotation_fields(ann: Annotation, exclude_annotation_fields: Set[str]):
    return json.dumps(
        {k: v for k, v in ann.asdict().items() if k not in exclude_annotation_fields},  # type: ignore
        sort_keys=True,
    )


def has_one_of_the_labels(ann: Annotation, label_field: str, labels: Collection[str]) -> bool:
    return getattr(ann, label_field) in labels


def has_this_label(ann: Annotation, label_field: str, label: str) -> bool:
    return getattr(ann, label_field) == label


class F1Metric(DocumentMetric):
    def __init__(
        self,
        layer: str,
        label_field: Optional[str] = None,
        labels: Optional[Collection[str]] = None,
        show_as_markdown: bool = False,
    ):
        super().__init__()
        self.layer = layer
        self.label_field = label_field
        self.show_as_markdown = show_as_markdown

        self.labels = set(labels or [])
        assert (
            "MICRO" not in self.labels and "MACRO" not in self.labels
        ), "labels cannot contain 'MICRO' or 'MACRO' because they are used to capture aggregated metrics"
        if self.label_field is None:
            assert (
                len(self.labels) == 0
            ), "can not calculate metrics per label without a provided label_field"

    def reset(self):
        self.counts = defaultdict(lambda: (0, 0, 0))

    def add_counts(self, counts: Tuple[int, int, int], label: str):
        self.counts[label] = (
            self.counts[label][0] + counts[0],
            self.counts[label][1] + counts[1],
            self.counts[label][2] + counts[2],
        )

    def _update(self, document: Document):
        new_counts = eval_counts_for_layer(
            document=document,
            layer=self.layer,
            annotation_filter=partial(
                has_one_of_the_labels, label_field=self.label_field, labels=self.labels
            )
            if self.label_field is not None
            else None,
        )
        self.add_counts(new_counts, label="MICRO")
        for label in self.labels:
            new_counts = eval_counts_for_layer(
                document=document,
                layer=self.layer,
                annotation_filter=partial(
                    has_this_label, label_field=self.label_field, label=label
                ),
            )
            self.add_counts(new_counts, label=label)

    def _compute(self) -> Dict[str, Dict[str, float]]:
        res = dict()
        if len(self.labels) > 0:
            res["MACRO"] = {"f1": 0.0, "p": 0.0, "r": 0.0}
        for label, counts in self.counts.items():
            tp, fp, fn = counts
            if tp == 0:
                p, r, f1 = 0.0, 0.0, 0.0
            else:
                p = tp / (tp + fp)
                r = tp / (tp + fn)
                f1 = 2 * p * r / (p + r)
            res[label] = {"f1": f1, "p": p, "r": r}
            if label in self.labels:
                res["MACRO"]["f1"] += f1 / len(self.labels)
                res["MACRO"]["p"] += p / len(self.labels)
                res["MACRO"]["r"] += r / len(self.labels)
        if self.show_as_markdown:
            logger.info(f"\n{self.layer}:\n{pd.DataFrame(res).round(3).T.to_markdown()}")
        return res
