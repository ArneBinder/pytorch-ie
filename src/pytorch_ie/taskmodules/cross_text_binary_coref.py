import copy
import logging
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
)

import torch
from pie_core import Annotation, TaskEncoding, TaskModule
from pie_core.utils.dictionary import list_of_dicts2dict_of_lists
from pie_documents.annotations import Span
from pie_documents.documents import TextPairDocumentWithLabeledSpansAndBinaryCorefRelations
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision, BinaryF1Score
from transformers import AutoTokenizer, BatchEncoding
from typing_extensions import TypeAlias

from pytorch_ie.taskmodules.common.mixins import RelationStatisticsMixin
from pytorch_ie.taskmodules.metrics import WrappedMetricWithPrepareFunction
from pytorch_ie.utils.document import SpanNotAlignedWithTokenException, get_aligned_token_span
from pytorch_ie.utils.window import get_window_around_slice

logger = logging.getLogger(__name__)

InputEncodingType: TypeAlias = Dict[str, Any]
TargetEncodingType: TypeAlias = Sequence[float]
DocumentType: TypeAlias = TextPairDocumentWithLabeledSpansAndBinaryCorefRelations

TaskEncodingType: TypeAlias = TaskEncoding[
    DocumentType,
    InputEncodingType,
    TargetEncodingType,
]


class TaskOutputType(TypedDict, total=False):
    score: float
    is_similar: bool


ModelInputType: TypeAlias = Dict[str, torch.Tensor]
ModelTargetType: TypeAlias = Dict[str, torch.Tensor]
ModelOutputType: TypeAlias = Dict[str, torch.Tensor]

TaskModuleType: TypeAlias = TaskModule[
    # _InputEncoding, _TargetEncoding, _TaskBatchEncoding, _ModelBatchOutput, _TaskOutput
    DocumentType,
    InputEncodingType,
    TargetEncodingType,
    Tuple[ModelInputType, Optional[ModelTargetType]],
    ModelTargetType,
    TaskOutputType,
]


class SpanDoesNotFitIntoAvailableWindow(Exception):
    def __init__(self, span):
        self.span = span


def _get_labels(model_output: ModelTargetType, label_threshold: float) -> torch.Tensor:
    return (model_output["scores"] > label_threshold).to(torch.int)


def _get_scores(model_output: ModelTargetType) -> torch.Tensor:
    return model_output["scores"]


S = TypeVar("S", bound=Span)


def shift_span(span: S, offset: int) -> S:
    return span.copy(start=span.start + offset, end=span.end + offset)


@TaskModule.register()
class CrossTextBinaryCorefTaskModule(RelationStatisticsMixin, TaskModuleType):
    """This taskmodule processes documents of type
    TextPairDocumentWithLabeledSpansAndBinaryCorefRelations in preparation for a
    SequencePairSimilarityModelWithPooler."""

    DOCUMENT_TYPE = DocumentType

    def __init__(
        self,
        tokenizer_name_or_path: str,
        similarity_threshold: float = 0.9,
        max_window: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.similarity_threshold = similarity_threshold
        self.max_window = max_window if max_window is not None else self.tokenizer.model_max_length
        self.available_window = self.max_window - self.tokenizer.num_special_tokens_to_add()
        self.num_special_tokens_before = len(self._get_special_tokens_before_input())

    def _get_special_tokens_before_input(self) -> List[int]:
        dummy_ids = self.tokenizer.build_inputs_with_special_tokens(token_ids_0=[-1])
        return dummy_ids[: dummy_ids.index(-1)]

    def encode(self, documents: Union[DocumentType, Iterable[DocumentType]], **kwargs):
        self.reset_statistics()
        result = super().encode(documents=documents, **kwargs)
        self.show_statistics()
        return result

    def truncate_encoding_around_span(
        self, encoding: BatchEncoding, char_span: Span
    ) -> Tuple[Dict[str, List[int]], Span]:
        input_ids = copy.deepcopy(encoding["input_ids"])

        token_span = get_aligned_token_span(encoding=encoding, char_span=char_span)

        # truncate input_ids and shift token_start and token_end
        if len(input_ids) > self.available_window:
            window_slice = get_window_around_slice(
                slice=(token_span.start, token_span.end),
                max_window_size=self.available_window,
                available_input_length=len(input_ids),
            )
            if window_slice is None:
                raise SpanDoesNotFitIntoAvailableWindow(span=token_span)
            window_start, window_end = window_slice
            input_ids = input_ids[window_start:window_end]
            token_span = shift_span(token_span, offset=-window_start)

        truncated_encoding = self.tokenizer.prepare_for_model(ids=input_ids)
        # shift indices because we added special tokens to the input_ids
        token_span = shift_span(token_span, offset=self.num_special_tokens_before)

        return truncated_encoding, token_span

    def encode_input(
        self,
        document: DocumentType,
        is_training: bool = False,
    ) -> Optional[Union[TaskEncodingType, Sequence[TaskEncodingType]]]:
        self.collect_all_relations(kind="available", relations=document.binary_coref_relations)
        tokenizer_kwargs = dict(
            padding=False,
            truncation=False,
            add_special_tokens=False,
        )
        encoding = self.tokenizer(text=document.text, **tokenizer_kwargs)
        encoding_pair = self.tokenizer(text=document.text_pair, **tokenizer_kwargs)

        task_encodings = []
        for coref_rel in document.binary_coref_relations:
            # TODO: This can miss instances if both texts are the same. We could check that
            #   coref_rel.head is in document.labeled_spans (same for the tail), but would this
            #   slow down the encoding?
            if not (
                coref_rel.head.target == document.text
                or coref_rel.tail.target == document.text_pair
            ):
                raise ValueError(
                    f"It is expected that coref relations go from (head) spans over 'text' "
                    f"to (tail) spans over 'text_pair', but this is not the case for this "
                    f"relation (i.e. it points into the other direction): {coref_rel.resolve()}"
                )
            try:
                current_encoding, token_span = self.truncate_encoding_around_span(
                    encoding=encoding, char_span=coref_rel.head
                )
                current_encoding_pair, token_span_pair = self.truncate_encoding_around_span(
                    encoding=encoding_pair, char_span=coref_rel.tail
                )
            except SpanNotAlignedWithTokenException as e:
                logger.warning(
                    f"Could not get token offsets for argument ({e.span}) of coref relation: "
                    f"{coref_rel.resolve()}. Skip it."
                )
                self.collect_relation(kind="skipped_args_not_aligned", relation=coref_rel)
                continue
            except SpanDoesNotFitIntoAvailableWindow as e:
                logger.warning(
                    f"Argument span [{e.span}] does not fit into available token window "
                    f"({self.available_window}). Skip it."
                )
                self.collect_relation(
                    kind="skipped_span_does_not_fit_into_window", relation=coref_rel
                )
                continue

            task_encodings.append(
                TaskEncoding(
                    document=document,
                    inputs={
                        "encoding": current_encoding,
                        "encoding_pair": current_encoding_pair,
                        "pooler_start_indices": token_span.start,
                        "pooler_end_indices": token_span.end,
                        "pooler_pair_start_indices": token_span_pair.start,
                        "pooler_pair_end_indices": token_span_pair.end,
                    },
                    metadata={"candidate_annotation": coref_rel},
                )
            )
            self.collect_relation("used", coref_rel)
        return task_encodings

    def encode_target(
        self,
        task_encoding: TaskEncoding[DocumentType, InputEncodingType, TargetEncodingType],
    ) -> Optional[TargetEncodingType]:
        return task_encoding.metadata["candidate_annotation"].score

    def collate(
        self,
        task_encodings: Sequence[
            TaskEncoding[DocumentType, InputEncodingType, TargetEncodingType]
        ],
    ) -> Tuple[ModelInputType, Optional[ModelTargetType]]:
        inputs_dict = list_of_dicts2dict_of_lists(
            [task_encoding.inputs for task_encoding in task_encodings]
        )

        inputs = {
            k: (
                self.tokenizer.pad(v, return_tensors="pt").data
                if k in ["encoding", "encoding_pair"]
                else torch.tensor(v)
            )
            for k, v in inputs_dict.items()
        }
        for k, v in inputs.items():
            if k.startswith("pooler_") and k.endswith("_indices"):
                inputs[k] = v.unsqueeze(-1)

        if not task_encodings[0].has_targets:
            return inputs, None
        targets = {
            "scores": torch.tensor([task_encoding.targets for task_encoding in task_encodings])
        }
        return inputs, targets

    def configure_model_metric(self, stage: str) -> MetricCollection:
        return MetricCollection(
            metrics={
                "continuous": WrappedMetricWithPrepareFunction(
                    metric=MetricCollection(
                        {
                            "auroc": BinaryAUROC(),
                            "avg-P": BinaryAveragePrecision(validate_args=False),
                            # "roc": BinaryROC(validate_args=False),
                            # "PRCurve": BinaryPrecisionRecallCurve(validate_args=False),
                            "f1": BinaryF1Score(threshold=self.similarity_threshold),
                        }
                    ),
                    prepare_function=_get_scores,
                ),
            }
        )

    def unbatch_output(self, model_output: ModelTargetType) -> Sequence[TaskOutputType]:
        is_similar = (model_output["scores"] > self.similarity_threshold).detach().cpu().tolist()
        scores = model_output["scores"].detach().cpu().tolist()
        result: List[TaskOutputType] = [
            {"is_similar": is_sim, "score": prob} for is_sim, prob in zip(is_similar, scores)
        ]
        return result

    def create_annotations_from_output(
        self,
        task_encoding: TaskEncoding[DocumentType, InputEncodingType, TargetEncodingType],
        task_output: TaskOutputType,
    ) -> Iterator[Tuple[str, Annotation]]:
        if task_output["is_similar"]:
            score = task_output["score"]
            new_coref_rel = task_encoding.metadata["candidate_annotation"].copy(score=score)
            yield "binary_coref_relations", new_coref_rel
