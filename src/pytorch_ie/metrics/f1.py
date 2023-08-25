import logging
from collections import defaultdict
from functools import partial
from typing import Callable, Collection, Dict, Hashable, Optional, Tuple

import pandas as pd

from pytorch_ie.core import Annotation, Document, DocumentMetric

logger = logging.getLogger(__name__)


def has_one_of_the_labels(ann: Annotation, label_field: str, labels: Collection[str]) -> bool:
    return getattr(ann, label_field) in labels


def has_this_label(ann: Annotation, label_field: str, label: str) -> bool:
    return getattr(ann, label_field) == label


class F1Metric(DocumentMetric):
    """Computes the (micro aggregated) F1 score for a given layer. If labels are provided,
    it also computes the F1 score for each label separately and the macro F1 score.

    Args:
        layer: The layer to compute the F1 score for.
        labels: If provided, calculate F1 score for each label.
        label_field: The field to use for the label. Defaults to "label".
        show_as_markdown: If True, logs the F1 score as markdown on the console when calling compute().
    """

    def __init__(
        self,
        layer: str,
        labels: Optional[Collection[str]] = None,
        label_field: str = "label",
        show_as_markdown: bool = False,
    ):
        super().__init__()
        self.layer = layer
        self.label_field = label_field
        self.show_as_markdown = show_as_markdown

        self.per_label = labels is not None
        self.labels = labels or []
        if self.per_label:
            if "MICRO" in self.labels or "MACRO" in self.labels:
                raise ValueError(
                    "labels cannot contain 'MICRO' or 'MACRO' because they are used to capture aggregated metrics"
                )
            if len(self.labels) == 0:
                raise ValueError("labels cannot be empty")

    def reset(self):
        self.counts = defaultdict(lambda: (0, 0, 0))

    def calculate_counts(
        self,
        document: Document,
        annotation_filter: Optional[Callable[[Annotation], bool]] = None,
    ) -> Tuple[int, int, int]:
        annotation_filter = annotation_filter or (lambda ann: True)
        predicted_annotations = {
            ann for ann in document[self.layer].predictions if annotation_filter(ann)
        }
        gold_annotations = {ann for ann in document[self.layer] if annotation_filter(ann)}
        tp = len([ann for ann in predicted_annotations & gold_annotations])
        fn = len([ann for ann in gold_annotations - predicted_annotations])
        fp = len([ann for ann in predicted_annotations - gold_annotations])
        return tp, fp, fn

    def add_counts(self, counts: Tuple[int, int, int], label: str):
        self.counts[label] = (
            self.counts[label][0] + counts[0],
            self.counts[label][1] + counts[1],
            self.counts[label][2] + counts[2],
        )

    def _update(self, document: Document):
        new_counts = self.calculate_counts(
            document=document,
            annotation_filter=partial(
                has_one_of_the_labels, label_field=self.label_field, labels=self.labels
            )
            if self.per_label
            else None,
        )
        self.add_counts(new_counts, label="MICRO")
        for label in self.labels:
            new_counts = self.calculate_counts(
                document=document,
                annotation_filter=partial(
                    has_this_label, label_field=self.label_field, label=label
                ),
            )
            self.add_counts(new_counts, label=label)

    def _compute(self) -> Dict[str, Dict[str, float]]:
        res = dict()
        if self.per_label:
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
