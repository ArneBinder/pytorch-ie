import logging
from collections import defaultdict
from typing import Callable, Dict, Optional, Tuple, Union

import pandas as pd

from pytorch_ie.core import Annotation, Document, DocumentMetric
from pytorch_ie.utils.hydra import resolve_target

logger = logging.getLogger(__name__)


class ConfusionMatrix(DocumentMetric):
    """Computes the confusion matrix for a given annotation layer that contains labeled annotations.

    Args:
        layer: The layer to compute the confusion matrix for.
        label_field: The field to use for the label. Defaults to "label".
        unassignable_label: The label to use for false negative annotations. Defaults to "UNASSIGNABLE".
        undetected_label: The label to use for false positive annotations. Defaults to "UNDETECTED".
        strict: If True, raises an error if a base annotation has multiple gold labels. If False, logs a warning.
        show_as_markdown: If True, logs the confusion matrix as markdown on the console when calling compute().
        annotation_processor: A callable that processes the annotations before calculating the confusion matrix.
    """

    def __init__(
        self,
        layer: str,
        label_field: str = "label",
        show_as_markdown: bool = False,
        unassignable_label: str = "UNASSIGNABLE",
        undetected_label: str = "UNDETECTED",
        strict: bool = True,
        annotation_processor: Optional[Union[Callable[[Annotation], Annotation], str]] = None,
    ):
        super().__init__()
        self.layer = layer
        self.label_field = label_field
        self.unassignable_label = unassignable_label
        self.undetected_label = undetected_label
        self.strict = strict
        self.show_as_markdown = show_as_markdown
        if isinstance(annotation_processor, str):
            annotation_processor = resolve_target(annotation_processor)
        self.annotation_processor = annotation_processor

    def reset(self):
        self.counts: Dict[Tuple[str, str], int] = defaultdict(int)

    def calculate_counts(
        self,
        document: Document,
        annotation_filter: Optional[Callable[[Annotation], bool]] = None,
        annotation_processor: Optional[Callable[[Annotation], Annotation]] = None,
    ) -> Dict[Tuple[str, str], int]:
        annotation_processor = annotation_processor or (lambda ann: ann)
        annotation_filter = annotation_filter or (lambda ann: True)
        predicted_annotations = {
            annotation_processor(ann)
            for ann in document[self.layer].predictions
            if annotation_filter(ann)
        }
        gold_annotations = {
            annotation_processor(ann) for ann in document[self.layer] if annotation_filter(ann)
        }
        base2gold = defaultdict(list)
        for ann in gold_annotations:
            base_ann_kwargs = {self.label_field: "DUMMY_LABEL"}
            base_ann = ann.copy(**base_ann_kwargs)
            base2gold[base_ann].append(ann)
        base2pred = defaultdict(list)
        for ann in predicted_annotations:
            base_ann_kwargs = {self.label_field: "DUMMY_LABEL"}
            base_ann = ann.copy(**base_ann_kwargs)
            base2pred[base_ann].append(ann)

        # (gold_label, pred_label) -> count
        counts: Dict[Tuple[str, str], int] = defaultdict(int)
        for base_ann in set(base2gold) | set(base2pred):
            gold_labels = [getattr(ann, self.label_field) for ann in base2gold[base_ann]]
            pred_labels = [getattr(ann, self.label_field) for ann in base2pred[base_ann]]

            if self.undetected_label in gold_labels:
                raise ValueError(
                    f"The gold annotation has the label '{self.undetected_label}' for undetected instances. "
                    f"Set a different undetected_label."
                )
            if self.unassignable_label in pred_labels:
                raise ValueError(
                    f"The predicted annotation has the label '{self.unassignable_label}' for unassignable predictions. "
                    f"Set a different unassignable_label."
                )

            if len(gold_labels) > 1:
                msg = f"The base annotation {base_ann} has multiple gold labels: {gold_labels}."
                if self.strict:
                    raise ValueError(msg)
                else:
                    logger.warning(msg + " Skip this base annotation.")
                    continue

            # use placeholder labels for empty gold or prediction labels
            if len(gold_labels) == 0:
                gold_labels.append(self.undetected_label)
            if len(pred_labels) == 0:
                pred_labels.append(self.unassignable_label)

            # main logic
            for gold_label in gold_labels:
                for pred_label in pred_labels:
                    counts[(gold_label, pred_label)] += 1

        return counts

    def add_counts(self, counts: Dict[Tuple[str, str], int]):
        for key, value in counts.items():
            self.counts[key] += value

    def _update(self, document: Document):
        new_counts = self.calculate_counts(
            document=document,
            annotation_processor=self.annotation_processor,
        )
        self.add_counts(new_counts)

    def _compute(self) -> Dict[str, Dict[str, int]]:

        res: Dict[str, Dict[str, int]] = defaultdict(dict)
        for gold_label, pred_label in sorted(self.counts):
            res[gold_label][pred_label] = self.counts[(gold_label, pred_label)]

        if self.show_as_markdown:
            res_df = pd.DataFrame(res).fillna(0)
            # index is prediction, columns is gold
            gold_labels = res_df.columns
            pred_labels = res_df.index

            # re-arrange index and columns: sort and put undetected_label and unassignable_label at the end
            gold_labels_sorted = sorted(
                [gold_label for gold_label in gold_labels if gold_label != self.undetected_label]
            )
            # re-add undetected_label at the end, if it was in the gold labels
            if self.undetected_label in gold_labels:
                gold_labels_sorted = gold_labels_sorted + [self.undetected_label]
            pred_labels_sorted = sorted(
                [pred_label for pred_label in pred_labels if pred_label != self.unassignable_label]
            )
            # re-add unassignable_label at the end, if it was in the pred labels
            if self.unassignable_label in pred_labels:
                pred_labels_sorted = pred_labels_sorted + [self.unassignable_label]
            res_df_sorted = res_df.loc[pred_labels_sorted, gold_labels_sorted]

            # transpose and show as markdown: index is now gold, columns is prediction
            logger.info(f"\n{self.layer}:\n{res_df_sorted.T.to_markdown()}")
        return res
