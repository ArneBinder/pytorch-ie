"""
workflow:
    document
        -> (input_encoding, target_encoding) -> task_encoding
            -> model_encoding -> model_output
        -> task_output
    -> document
"""

import logging
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from pie_core import TaskEncoding, TaskModule
from transformers import AutoTokenizer
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import BatchEncoding, TruncationStrategy
from typing_extensions import TypeAlias

from pytorch_ie.annotations import LabeledSpan, MultiLabeledSpan, Span
from pytorch_ie.documents import (
    TextDocument,
    TextDocumentWithLabeledSpans,
    TextDocumentWithLabeledSpansAndLabeledPartitions,
    TextDocumentWithLabeledSpansAndSentences,
)
from pytorch_ie.models.transformer_span_classification import ModelOutputType, ModelStepInputType

InputEncodingType: TypeAlias = BatchEncoding
TargetEncodingType: TypeAlias = Sequence[Tuple[int, int, int]]

TaskEncodingType: TypeAlias = TaskEncoding[
    TextDocument,
    InputEncodingType,
    TargetEncodingType,
]
TaskOutputType: TypeAlias = Dict[str, Any]

TaskModuleType: TypeAlias = TaskModule[
    TextDocument,
    InputEncodingType,
    TargetEncodingType,
    ModelStepInputType,
    ModelOutputType,
    TaskOutputType,
]


logger = logging.getLogger(__name__)


@TaskModule.register()
class TransformerSpanClassificationTaskModule(TaskModuleType):
    PREPARED_ATTRIBUTES = ["label_to_id"]

    def __init__(
        self,
        tokenizer_name_or_path: str,
        entity_annotation: str = "labeled_spans",
        single_sentence: bool = False,
        sentence_annotation: str = "sentences",
        padding: Union[bool, str, PaddingStrategy] = True,
        truncation: Union[bool, str, TruncationStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        label_pad_token_id: int = -100,
        label_to_id: Optional[Dict[str, int]] = None,
        multi_label: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters()

        if multi_label:
            raise NotImplementedError(
                "Multi-label classification (multi_label=True) is not supported yet."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.entity_annotation = entity_annotation
        self.single_sentence = single_sentence
        self.sentence_annotation = sentence_annotation
        if label_to_id is not None:
            self.label_to_id = label_to_id
        self.padding = padding
        self.truncation = truncation
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id
        self.multi_label = multi_label

    @property
    def document_type(self) -> Optional[Type[TextDocument]]:
        dt: Type[TextDocument]
        if self.single_sentence:
            dt = TextDocumentWithLabeledSpansAndSentences
        else:
            dt = TextDocumentWithLabeledSpans

        if self.entity_annotation == "labeled_spans":
            return dt
        else:
            logger.warning(
                f"entity_annotation={self.entity_annotation} is "
                f"not the default value ('labeled_spans'), so the taskmodule {type(self).__name__} can not request "
                f"the usual document type ({dt.__name__}) for auto-conversion because this has the bespoken default "
                f"value as layer name instead of the provided one."
            )
            return None

    def _prepare(self, documents: Sequence[TextDocument]) -> None:
        labels: Set[str] = set()
        for document in documents:
            entities: Union[Sequence[LabeledSpan], Sequence[MultiLabeledSpan]] = document[
                self.entity_annotation
            ]

            for entity in entities:
                if self.multi_label and not isinstance(entity, MultiLabeledSpan):
                    raise ValueError("Spans must be MultiLabeledSpan if multi_label=True.")

                if not self.multi_label and not isinstance(entity, LabeledSpan):
                    raise ValueError("Spans must be LabeledSpan if multi_label=False.")

                if self.multi_label:
                    labels.update(entity.label)
                else:
                    labels.add(entity.label)

        self.label_to_id = {"O": 0}
        label_id = 1
        for label in sorted(labels):
            self.label_to_id[label] = label_id
            label_id += 1

    def _post_prepare(self) -> None:
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

    def encode_input(
        self,
        document: TextDocument,
    ) -> Optional[Union[TaskEncodingType, Sequence[TaskEncodingType]]]:
        partitions: Sequence[Span]
        if self.single_sentence:
            partitions = document[self.sentence_annotation]
        else:
            partitions = [Span(start=0, end=len(document.text))]

        task_encodings: List[TaskEncoding] = []
        for partition_idx, partition in enumerate(partitions):
            inputs = self.tokenizer(
                document.text[partition.start : partition.end],
                padding=False,
                truncation=self.truncation,
                max_length=self.max_length,
                is_split_into_words=False,
                return_offsets_mapping=True,
                return_special_tokens_mask=True,
            )

            metadata = {
                "offset_mapping": inputs.pop("offset_mapping"),
                "special_tokens_mask": inputs.pop("special_tokens_mask"),
            }

            if self.single_sentence:
                metadata["partition_idx"] = partition_idx

            task_encodings.append(
                TaskEncoding(
                    document=document,
                    inputs=inputs,
                    metadata=metadata,
                )
            )

        return task_encodings

    def encode_target(
        self,
        task_encoding: TaskEncodingType,
    ) -> TargetEncodingType:
        # inputs is a transformers BatchEncoding object
        document = task_encoding.document
        inputs = task_encoding.inputs
        metadata = task_encoding.metadata

        targets: List[Tuple[int, int, int]] = []
        entities: Sequence[LabeledSpan] = document[self.entity_annotation]
        if self.single_sentence:
            partition_idx = metadata["partition_idx"]
            partitions: Sequence[Span] = document[self.sentence_annotation]
            assert (
                partitions is not None
            ), f"document has no span annotations with name '{self.sentence_annotation}'"
            partition = partitions[partition_idx]

            for entity in entities:
                if entity.start < partition.start or entity.end > partition.end:
                    continue

                entity_start = entity.start - partition.start
                entity_end = entity.end - partition.start

                start_idx = inputs.char_to_token(entity_start)
                end_idx = inputs.char_to_token(entity_end - 1)

                # TODO: remove this is if case
                if start_idx is None or end_idx is None:
                    logger.warning(
                        f"Entity annotation does not start or end with a token, it will be skipped: {entity}"
                    )
                    continue

                targets.append((start_idx, end_idx, self.label_to_id[entity.label]))
        else:
            for entity in entities:
                start_idx = inputs.char_to_token(entity.start)
                end_idx = inputs.char_to_token(entity.end - 1)
                targets.append((start_idx, end_idx, self.label_to_id[entity.label]))

        return targets

    def unbatch_output(self, model_output: ModelOutputType) -> Sequence[TaskOutputType]:
        logits = model_output["logits"]
        probs = F.softmax(logits, dim=-1).detach().cpu().float().numpy()
        label_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()

        start_indices = model_output["start_indices"].detach().cpu().numpy()
        end_indices = model_output["end_indices"].detach().cpu().numpy()
        batch_indices = model_output["batch_indices"].detach().cpu().numpy()

        tags: List[List[Tuple[str, Tuple[int, int]]]] = [[] for _ in np.unique(batch_indices)]
        probabilities: List[List[float]] = [[] for _ in np.unique(batch_indices)]
        for start, end, batch_idx, label_id, prob in zip(
            start_indices, end_indices, batch_indices, label_ids, probs
        ):
            label = self.id_to_label[label_id]
            if label != "O":
                tags[batch_idx].append((label, (start, end)))
                probabilities[batch_idx].append(prob[label_id])

        return [{"tags": t, "probabilities": p} for t, p in zip(tags, probabilities)]

    def create_annotations_from_output(
        self,
        task_encoding: TaskEncodingType,
        task_output: TaskOutputType,
    ) -> Iterator[Tuple[str, Union[LabeledSpan, MultiLabeledSpan]]]:
        document = task_encoding.document
        metadata = task_encoding.metadata

        if self.single_sentence:
            partitions: Sequence[Span] = document[self.sentence_annotation]
            sentence = partitions[metadata["sentence_index"]]

            spans = task_output["tags"]
            probabilities = task_output["probabilities"]
            for (label, (start, end)), probability in zip(spans, probabilities):
                yield (
                    self.entity_annotation,
                    LabeledSpan(
                        start=sentence.start + metadata["offset_mapping"][start][0],
                        end=sentence.start + metadata["offset_mapping"][end][1],
                        label=label,
                        score=float(probability),
                    ),
                )
        else:
            spans = task_output["tags"]
            probabilities = task_output["probabilities"]
            for (label, (start, end)), probability in zip(spans, probabilities):
                yield (
                    self.entity_annotation,
                    LabeledSpan(
                        start=metadata["offset_mapping"][start][0],
                        end=metadata["offset_mapping"][end][1],
                        label=label,
                        score=float(probability),
                    ),
                )

    def collate(self, task_encodings: Sequence[TaskEncodingType]) -> ModelStepInputType:
        input_features = [task_encoding.inputs for task_encoding in task_encodings]

        inputs: Dict[str, torch.Tensor] = self.tokenizer.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if not task_encodings[0].has_targets:
            return inputs, None

        targets: Sequence[TargetEncodingType] = [
            task_encoding.targets for task_encoding in task_encodings
        ]

        inputs = {k: torch.tensor(v, dtype=torch.int64) for k, v in inputs.items()}

        return inputs, targets
