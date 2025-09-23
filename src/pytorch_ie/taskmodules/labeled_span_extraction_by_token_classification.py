"""
workflow:
    Document
        -> (InputEncoding, TargetEncoding) -> TaskEncoding
            -> ModelStepInputType -> ModelBatchOutput
        -> TaskOutput
    -> Document
"""

import logging
from functools import partial
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypedDict,
    Union,
)

import torch
from pie_core import AnnotationLayer, TaskEncoding, TaskModule
from pie_core.utils.dictionary import list_of_dicts2dict_of_lists
from pie_documents.annotations import LabeledSpan
from pie_documents.document.processing import token_based_document_to_text_based
from pie_documents.documents import (
    TextBasedDocument,
    TextDocumentWithLabeledSpans,
    TextDocumentWithLabeledSpansAndLabeledPartitions,
    TokenDocumentWithLabeledSpans,
    TokenDocumentWithLabeledSpansAndLabeledPartitions,
)
from pie_documents.utils.sequence_tagging import tag_sequence_to_token_spans
from tokenizers import Encoding
from torchmetrics import F1Score, Metric, MetricCollection, Precision, Recall
from transformers import AutoTokenizer
from typing_extensions import TypeAlias

from pytorch_ie.models.simple_token_classification import InputType as ModelInputType
from pytorch_ie.models.simple_token_classification import TargetType as ModelTargetType
from pytorch_ie.taskmodules.metrics import (
    PrecisionRecallAndF1ForLabeledAnnotations,
    WrappedMetricWithPrepareFunction,
)
from pytorch_ie.utils.document import tokenize_document

DocumentType: TypeAlias = TextBasedDocument

InputEncodingType: TypeAlias = Encoding
TargetEncodingType: TypeAlias = Sequence[int]
TaskEncodingType: TypeAlias = TaskEncoding[
    DocumentType,
    InputEncodingType,
    TargetEncodingType,
]
ModelStepInputType: TypeAlias = Tuple[
    ModelInputType,
    Optional[ModelTargetType],
]
ModelOutputType: TypeAlias = ModelTargetType


class TaskOutputType(TypedDict, total=False):
    labels: torch.LongTensor
    probabilities: torch.FloatTensor


TaskModuleType: TypeAlias = TaskModule[
    DocumentType,
    InputEncodingType,
    TargetEncodingType,
    ModelStepInputType,
    ModelOutputType,
    TaskOutputType,
]

logger = logging.getLogger(__name__)


def _get_label_ids_from_model_output(
    model_output: ModelTargetType,
) -> torch.LongTensor:
    result = model_output["labels"]
    assert isinstance(result, torch.LongTensor)
    return result


def unbatch_and_decode_annotations(
    model_output: ModelOutputType,
    taskmodule: "LabeledSpanExtractionByTokenClassificationTaskModule",
) -> List[Sequence[LabeledSpan]]:
    task_outputs = taskmodule.unbatch_output(model_output)
    annotations = [
        taskmodule.decode_annotations(task_output)["labeled_spans"] for task_output in task_outputs
    ]
    return annotations


@TaskModule.register()
class LabeledSpanExtractionByTokenClassificationTaskModule(TaskModuleType):
    """Taskmodule for span prediction (e.g. NER) as token classification.

    This taskmodule expects the input documents to be of TextBasedDocument with an annotation layer of
    labeled spans (e.g. TextDocumentWithLabeledSpans). The text is tokenized using the provided tokenizer and
    the labels are converted to BIO tags.

    To handle long documents, the text can be windowed using the respective parameters for the tokenizer,
    i.e. max_length (and stride). Note, that this requires to set return_overflowing_tokens=True, otherwise just
    the first window of input tokens is considered. The windowing is done in a way that the spans are not split
    across windows. If a span is split across windows, it is ignored during training and evaluation. Thus, if you
    have long spans in your data, it is recommended to set a stride that is as large as the average span length
    to avoid missing many spans.

    If a partition annotation is provided, the taskmodule expects the input documents to be of
    TextBasedDocument with two annotation layers of labeled spans, one for the spans and one for the partitions
    (e.g. TextDocumentWithLabeledSpansAndLabeledPartitions). Then, the text is tokenized and fed to the model
    individually per partition (e.g. per sentence). This is useful for long documents that can not be processed
    by the model as a whole, but where a natural partitioning exists (e.g. sentences or paragraphs) and, thus,
    windowing is not necessary (or a combination of both can be used).

    If labels are not provided, they are collected from the data during the prepare() step. If provided, they act as
    whitelist, i.e. spans with labels that are not in the labels are ignored during training and evaluation.

    Args:
        tokenizer_name_or_path: Name or path of the HuggingFace tokenizer to use.
        span_annotation: Name of the annotation layer that contains the labeled spans. Default: "labeled_spans".
        partition_annotation: Name of the annotation layer that contains the labeled partitions. If provided, the
            text is tokenized individually per partition. Default: None.
        label_pad_id: ID of the padding tag label. The model should ignore this for training. Default: -100.
        labels: List of labels to use. If not provided, the labels are collected from the labeled span annotations
            in the data during the prepare() step. Default: None.
        include_ill_formed_predictions: Whether to include ill-formed predictions in the output. If False, the
            predictions are corrected to be well-formed. Default: True.
        tokenize_kwargs: Keyword arguments to pass to the tokenizer during tokenization. Default: None.
        pad_kwargs: Keyword arguments to pass to the tokenizer during padding. Note, that this is used to pad the
            token ids *and* the tag ids, if available (i.e. during training or evaluation). Default: None.
        combine_token_scores_method: Method to combine the token scores to a span score. Options are "mean", "max",
            "min", and "product". Default: "mean".
        log_precision_recall_metrics: Whether to log precision and recall metrics (in addition to F1) for the
            spans. Default: True.
    """

    # list of attribute names that need to be set by _prepare()
    PREPARED_ATTRIBUTES: List[str] = ["labels"]

    def __init__(
        self,
        tokenizer_name_or_path: str,
        span_annotation: str = "labeled_spans",
        partition_annotation: Optional[str] = None,
        label_pad_id: int = -100,
        labels: Optional[List[str]] = None,
        include_ill_formed_predictions: bool = True,
        tokenize_kwargs: Optional[Dict[str, Any]] = None,
        pad_kwargs: Optional[Dict[str, Any]] = None,
        combine_token_scores_method: str = "mean",
        log_precision_recall_metrics: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.span_annotation = span_annotation
        self.partition_annotation = partition_annotation
        self.labels = labels
        self.label_pad_id = label_pad_id
        self.include_ill_formed_predictions = include_ill_formed_predictions
        self.tokenize_kwargs = tokenize_kwargs or {}
        self.pad_kwargs = pad_kwargs or {}
        self.log_precision_recall_metrics = log_precision_recall_metrics
        self.combine_token_scores_method = combine_token_scores_method

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    @property
    def document_type(self) -> Optional[Type[TextBasedDocument]]:
        dt: Type[TextBasedDocument]
        errors = []
        if self.span_annotation != "labeled_spans":
            errors.append(
                f"span_annotation={self.span_annotation} is not the default value ('labeled_spans')"
            )
        if self.partition_annotation is None:
            dt = TextDocumentWithLabeledSpans
        else:
            if self.partition_annotation != "labeled_partitions":
                errors.append(
                    f"partition_annotation={self.partition_annotation} is not the default value "
                    f"('labeled_partitions')"
                )
            dt = TextDocumentWithLabeledSpansAndLabeledPartitions

        if len(errors) == 0:
            return dt
        else:
            logger.warning(
                f"{' and '.join(errors)}, so the taskmodule {type(self).__name__} can not request "
                f"the usual document type ({dt.__name__}) for auto-conversion because this has the bespoken default "
                f"value as layer name(s) instead of the provided one(s)."
            )
            return None

    def get_span_layer(self, document: DocumentType) -> AnnotationLayer[LabeledSpan]:
        return document[self.span_annotation]

    def _prepare(self, documents: Sequence[DocumentType]) -> None:
        # collect all possible labels
        labels: Set[str] = set()
        for document in documents:
            spans: AnnotationLayer[LabeledSpan] = self.get_span_layer(document)

            for span in spans:
                labels.add(span.label)

        self.labels = sorted(labels)
        logger.info(f"Collected {len(self.labels)} labels from the data: {self.labels}")

    def _post_prepare(self):
        # create the real token labels (BIO scheme) from the labels
        self.label_to_id = {"O": 0}
        current_id = 1
        for label in sorted(self.labels):
            for prefix in ["B", "I"]:
                self.label_to_id[f"{prefix}-{label}"] = current_id
                current_id += 1

        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

    def encode_input(
        self,
        document: TextBasedDocument,
    ) -> Optional[Union[TaskEncodingType, Sequence[TaskEncodingType]]]:
        if self.partition_annotation is None:
            tokenized_document_type = TokenDocumentWithLabeledSpans
            casted_document_type = TextDocumentWithLabeledSpans
            field_mapping = {self.span_annotation: "labeled_spans"}
        else:
            tokenized_document_type = TokenDocumentWithLabeledSpansAndLabeledPartitions
            casted_document_type = TextDocumentWithLabeledSpansAndLabeledPartitions
            field_mapping = {
                self.span_annotation: "labeled_spans",
                self.partition_annotation: "labeled_partitions",
            }
        casted_document = document.as_type(casted_document_type, field_mapping=field_mapping)
        tokenized_docs = tokenize_document(
            casted_document,
            tokenizer=self.tokenizer,
            result_document_type=tokenized_document_type,
            partition_layer=(
                "labeled_partitions" if self.partition_annotation is not None else None
            ),
            strict_span_conversion=False,
            **self.tokenize_kwargs,
        )

        task_encodings: List[TaskEncodingType] = []
        for tokenized_doc in tokenized_docs:
            task_encodings.append(
                TaskEncoding(
                    document=document,
                    inputs=tokenized_doc.metadata["tokenizer_encoding"],
                    metadata={"tokenized_document": tokenized_doc},
                )
            )

        return task_encodings

    def encode_target(
        self,
        task_encoding: TaskEncodingType,
    ) -> Optional[TargetEncodingType]:
        metadata = task_encoding.metadata
        tokenized_document = metadata["tokenized_document"]
        tokenizer_encoding: Encoding = tokenized_document.metadata["tokenizer_encoding"]

        tag_sequence = [
            None if tokenizer_encoding.special_tokens_mask[j] else "O"
            for j in range(len(tokenizer_encoding.ids))
        ]
        if self.labels is None:
            raise ValueError(
                "'labels' must be set before calling encode_target(). Was prepare() called on the taskmodule?"
            )
        sorted_spans = sorted(tokenized_document.labeled_spans, key=lambda s: (s.start, s.end))
        for span in sorted_spans:
            if span.label not in self.labels:
                continue
            start = span.start
            end = span.end
            if any(tag != "O" for tag in tag_sequence[start:end]):
                logger.warning(f"tag already assigned (current span has an overlap: {span}).")
                continue

            tag_sequence[start] = f"B-{span.label}"
            for j in range(start + 1, end):
                tag_sequence[j] = f"I-{span.label}"

        targets = [
            self.label_to_id[tag] if tag is not None else self.label_pad_id for tag in tag_sequence
        ]

        return targets

    def collate(self, task_encodings: Sequence[TaskEncodingType]) -> ModelStepInputType:
        input_encodings = [
            {
                "input_ids": task_encoding.inputs.ids,
                "attention_mask": task_encoding.inputs.attention_mask,
                "special_tokens_mask": task_encoding.inputs.special_tokens_mask,
            }
            for task_encoding in task_encodings
        ]
        inputs = self.tokenizer.pad(
            list_of_dicts2dict_of_lists(input_encodings), return_tensors="pt", **self.pad_kwargs
        )

        if not task_encodings[0].has_targets:
            return inputs, None

        tag_ids = [task_encoding.targets for task_encoding in task_encodings]
        targets = self.tokenizer.pad(
            {"input_ids": tag_ids}, return_tensors="pt", **self.pad_kwargs
        )["input_ids"]

        # set the padding label to the label_pad_token_id
        pad_mask = inputs["input_ids"] == self.tokenizer.pad_token_id
        targets[pad_mask] = self.label_pad_id

        return inputs, {"labels": targets}

    def unbatch_output(self, model_output: ModelOutputType) -> Sequence[TaskOutputType]:
        labels = model_output["labels"]
        probabilities = model_output.get("probabilities", None)
        batch_size = labels.shape[0]
        task_outputs: List[TaskOutputType] = []
        for batch_idx in range(batch_size):
            batch_labels = labels[batch_idx]
            assert isinstance(batch_labels, torch.LongTensor)
            task_output: TaskOutputType = {"labels": batch_labels}
            if probabilities is not None:
                batch_probabilities = probabilities[batch_idx]
                assert isinstance(batch_probabilities, torch.FloatTensor)
                task_output["probabilities"] = batch_probabilities
            task_outputs.append(task_output)
        return task_outputs

    def decode_annotations(self, encoding: TaskOutputType) -> Dict[str, Sequence[LabeledSpan]]:
        labels = encoding["labels"]
        tag_sequence = [
            "O" if tag_id == self.label_pad_id else self.id_to_label[tag_id]
            for tag_id in labels.tolist()
        ]
        labeled_spans: List[LabeledSpan] = []
        for label, (start, end_inclusive) in tag_sequence_to_token_spans(
            tag_sequence,
            coding_scheme="IOB2",
            include_ill_formed=self.include_ill_formed_predictions,
        ):
            end = end_inclusive + 1
            # do not set the score if the probabilities are not available
            annotation_kwargs = {}
            if encoding.get("probabilities") is not None:
                span_probabilities = encoding["probabilities"][start:end]
                span_label_ids = labels[start:end]
                # get the probabilities at the label indices
                span_label_probs = torch.stack(
                    [span_probabilities[i, l] for i, l in enumerate(span_label_ids)]
                )
                if self.combine_token_scores_method == "mean":
                    # use mean probability of the span as score
                    annotation_kwargs["score"] = span_label_probs.mean().item()
                elif self.combine_token_scores_method == "max":
                    # use max probability of the span as score
                    annotation_kwargs["score"] = span_label_probs.max().item()
                elif self.combine_token_scores_method == "min":
                    # use min probability of the span as score
                    annotation_kwargs["score"] = span_label_probs.min().item()
                elif self.combine_token_scores_method == "product":
                    # use product of probabilities of the span as score
                    annotation_kwargs["score"] = span_label_probs.prod().item()
                else:
                    raise ValueError(
                        f"combine_token_scores_method={self.combine_token_scores_method} is not supported."
                    )
            labeled_span = LabeledSpan(label=label, start=start, end=end, **annotation_kwargs)
            labeled_spans.append(labeled_span)
        return {"labeled_spans": labeled_spans}

    def create_annotations_from_output(
        self,
        task_encoding: TaskEncodingType,
        task_output: TaskOutputType,
    ) -> Iterator[Tuple[str, LabeledSpan]]:
        tokenized_document = task_encoding.metadata["tokenized_document"]
        decoded_annotations = self.decode_annotations(task_output)

        # Note: token_based_document_to_text_based() does not yet consider predictions, so we need to clear
        # the main annotations and attach the predictions to that
        for layer_name, annotations in decoded_annotations.items():
            tokenized_document[layer_name].clear()
            for annotation in annotations:
                tokenized_document[layer_name].append(annotation)

        # we can not use self.document_type here because that may be None if self.span_annotation or
        # self.partition_annotation is not the default value
        document_type = (
            TextDocumentWithLabeledSpansAndLabeledPartitions
            if self.partition_annotation
            else TextDocumentWithLabeledSpans
        )
        untokenized_document: Union[
            TextDocumentWithLabeledSpans, TextDocumentWithLabeledSpansAndLabeledPartitions
        ] = token_based_document_to_text_based(
            tokenized_document, result_document_type=document_type
        )

        for span in untokenized_document.labeled_spans:
            # need to copy the span because it can be attached to only one document
            yield self.span_annotation, span.copy()

    def configure_model_metric(self, stage: str) -> Union[Metric, MetricCollection]:
        common_metric_kwargs: dict[str, Any] = {
            "num_classes": len(self.label_to_id),
            "task": "multiclass",
            "ignore_index": self.label_pad_id,
        }
        token_scores = MetricCollection(
            {
                "token/macro/f1": WrappedMetricWithPrepareFunction(
                    metric=F1Score(average="macro", **common_metric_kwargs),
                    prepare_function=_get_label_ids_from_model_output,
                ),
                "token/micro/f1": WrappedMetricWithPrepareFunction(
                    metric=F1Score(average="micro", **common_metric_kwargs),
                    prepare_function=_get_label_ids_from_model_output,
                ),
                "token/macro/precision": WrappedMetricWithPrepareFunction(
                    metric=Precision(average="macro", **common_metric_kwargs),
                    prepare_function=_get_label_ids_from_model_output,
                ),
                "token/macro/recall": WrappedMetricWithPrepareFunction(
                    metric=Recall(average="macro", **common_metric_kwargs),
                    prepare_function=_get_label_ids_from_model_output,
                ),
                "token/micro/precision": WrappedMetricWithPrepareFunction(
                    metric=Precision(average="micro", **common_metric_kwargs),
                    prepare_function=_get_label_ids_from_model_output,
                ),
                "token/micro/recall": WrappedMetricWithPrepareFunction(
                    metric=Recall(average="micro", **common_metric_kwargs),
                    prepare_function=_get_label_ids_from_model_output,
                ),
            }
        )

        span_scores = PrecisionRecallAndF1ForLabeledAnnotations(
            flatten_result_with_sep="/",
            prefix="span/",
            return_recall_and_precision=self.log_precision_recall_metrics,
        )
        span_scores_wrapped = WrappedMetricWithPrepareFunction(
            metric=span_scores,
            prepare_function=partial(unbatch_and_decode_annotations, taskmodule=self),
            prepare_does_unbatch=True,
        )
        # TODO: mypy complains that MetricCollection does not accept a list containing MetricCollection(s), but it works fine
        result = MetricCollection([token_scores, span_scores_wrapped])  # type: ignore
        return result
