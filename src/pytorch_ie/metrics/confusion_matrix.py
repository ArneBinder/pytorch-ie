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
        missed_label: The label to use for missed annotations. Defaults to "MISSED".
        strict: If True, raises an error if a base annotation has multiple gold labels. If False, logs a warning.
        show_as_markdown: If True, logs the confusion matrix as markdown on the console when calling compute().
        annotation_processor: A callable that processes the annotations before calculating the confusion matrix.
    """

    def __init__(
        self,
        layer: str,
        label_field: str = "label",
        show_as_markdown: bool = False,
        missed_label: str = "MISSED",
        strict: bool = True,
        annotation_processor: Optional[Union[Callable[[Annotation], Annotation], str]] = None,
    ):
        super().__init__()
        self.layer = layer
        self.label_field = label_field
        self.missed_label = missed_label
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

            if self.missed_label in gold_labels:
                raise ValueError(
                    f"The gold annotation has the na_label='{self.missed_label}' as label. "
                    f"Set a different na_label."
                )
            if self.missed_label in pred_labels:
                raise ValueError(
                    f"The predicted annotation has the na_label='{self.missed_label}' as label. "
                    f"Set a different na_label."
                )

            if len(gold_labels) > 1:
                msg = f"The base annotation {base_ann} has multiple gold labels: {gold_labels}."
                if self.strict:
                    raise ValueError(msg)
                else:
                    logger.warning(msg + " Skip this base annotation.")
                    continue

            # TODO: is this logic correct?
            if len(gold_labels) == 0:
                gold_labels.append(self.missed_label)
            if len(pred_labels) == 0:
                pred_labels.append(self.missed_label)
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
            # sort index and columns
            columns_sorted = sorted([col for col in res_df.columns if col != self.missed_label])
            if self.missed_label in res_df.columns:
                columns_sorted = columns_sorted + [self.missed_label]
            index_sorted = sorted([idx for idx in res_df.index if idx != self.missed_label])
            if self.missed_label in res_df.index:
                index_sorted = index_sorted + [self.missed_label]
            res_df_sorted = res_df.loc[index_sorted, columns_sorted]

            # show as markdown: index is gold, columns is prediction
            logger.info(f"\n{self.layer}:\n{res_df_sorted.T.to_markdown()}")
        return res
