import logging
from dataclasses import dataclass, fields
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple, TypeVar, Union

import numpy as np
import torch
from torch import softmax
from transformers import LayoutXLMProcessor
from transformers.file_utils import PaddingStrategy
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.tokenization_utils_base import TruncationStrategy
from typing_extensions import TypeAlias

from pytorch_ie.annotations import Label, OcrLabeledSpan
from pytorch_ie.core import TaskEncoding, TaskModule
from pytorch_ie.documents import OcrDocumentWithEntities
from pytorch_ie.utils.span import bio_tags_to_spans

logger = logging.getLogger(__name__)


@dataclass
class DataClassWithAsDict:
    def asdict(self, exclude: Optional[List[str]] = None):
        exclude_set = set(exclude or [])
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name not in exclude_set
        }


@dataclass
class InputEncoding(DataClassWithAsDict):
    input_ids: List[int]
    attention_mask: List[int]
    bbox: List[List[int]]
    image: List[List[List[int]]]
    label_positions: List[List[int]]
    chunk: Tuple[int, int]
    id: Optional[str] = None


@dataclass
class TargetEncoding(DataClassWithAsDict):
    word_labels: List[int]


@dataclass
class TaskOutput(DataClassWithAsDict):
    tags: List[str]
    probabilities: np.ndarray


"""
workflow:
    document
        -> (input_encoding, target_encoding) -> task_encoding
            -> model_encoding -> model_output
        -> task_output
    -> document
"""


# Define task specific input and output types
DocumentType: TypeAlias = OcrDocumentWithEntities
InputEncodingType: TypeAlias = InputEncoding
TargetEncodingType: TypeAlias = TargetEncoding
ModelEncodingType: TypeAlias = Tuple[Dict[str, Any], Optional[torch.Tensor]]
ModelOutputType: TypeAlias = TokenClassifierOutput
TaskOutputType: TypeAlias = TaskOutput

# This should be the same for all taskmodules
TaskEncodingType: TypeAlias = TaskEncoding[DocumentType, InputEncodingType, TargetEncodingType]
TaskModuleType: TypeAlias = TaskModule[
    DocumentType,
    InputEncodingType,
    TargetEncodingType,
    ModelEncodingType,
    ModelOutputType,
    TaskOutputType,
]


T_pad_entry = TypeVar("T_pad_entry")


def pad_sequences(
    sequences: List[List[T_pad_entry]],
    pad_entry: T_pad_entry,
    sequence_length: int,
    padding_side: str,
) -> List[List[T_pad_entry]]:
    if padding_side == "right":
        return [
            sequence + [pad_entry] * (sequence_length - len(sequence)) for sequence in sequences
        ]
    else:
        return [
            [pad_entry] * (sequence_length - len(sequence)) + sequence for sequence in sequences
        ]


def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]


@TaskModule.register()
class MultiModalSequenceTaggingTaskModule(TaskModuleType):
    def __init__(
        self,
        processor_name_or_path: str,
        label_encoding: str = "IOB2",
        exclude_labels: Optional[List] = None,
        label_pad_token_id: int = -100,
        padding: Union[bool, str, PaddingStrategy] = True,
        truncation: Union[bool, str, TruncationStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        label_to_id: Optional[Dict[str, int]] = None,
        processor_kwargs: Optional[Dict[str, Any]] = None,
        include_ill_formed_predictions: bool = True,
        **kwargs,
    ) -> None:
        # Important: Remaining keyword arguments need to be passed to super.
        super().__init__(**kwargs)
        # Save all passed arguments. They will be available via self._config().
        self.save_hyperparameters()

        self.processor_kwargs = processor_kwargs or {}
        self.processor = LayoutXLMProcessor.from_pretrained(
            processor_name_or_path, **self.processor_kwargs
        )

        self.label_encoding = label_encoding
        self.include_ill_formed_predictions = include_ill_formed_predictions
        self.exclude_labels = set(exclude_labels if exclude_labels is not None else [])

        # some tokenization and padding parameters
        self.label_pad_token_id = label_pad_token_id
        self.truncation = truncation
        self.max_length = max_length
        self.padding = padding
        self.pad_to_multiple_of = pad_to_multiple_of

        # this will be prepared from the data or loaded from the config
        self.label_to_id = label_to_id or {}
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

    def _config(self) -> Dict[str, Any]:
        """
        Add config entries. The config will be dumped when calling save_pretrained().
        Entries of the config will be passed to the constructor of this taskmodule when
        loading it with from_pretrained().
        """
        # add the label-to-id mapping to the config
        config = super()._config()
        config["label_to_id"] = self.label_to_id
        return config

    def prepare(self, documents: Sequence[DocumentType]) -> None:
        """
        Prepare the task module with training documents, e.g. collect all possible labels.
        """

        # Don't do anything if we directly created a prepared taskmodule. This may be useful for very large
        # datasets where it is not reasonable to scan them before training.
        if len(self.label_to_id) > 0:
            logger.warning(
                f"It looks like the taskmodule is already prepared since label_to_id contains entries, "
                f"so they are not collected again. label_to_id = {str(self.label_to_id)}"
            )
        else:
            # create the label-to-id mapping
            labels: Set[str] = set()
            _exclude_labels = self.exclude_labels | {None}
            for document in documents:
                labels.update(
                    entity.label
                    for entity in document.entities
                    if entity.label not in _exclude_labels
                )
            if self.label_encoding == "IOB2":
                self.label_to_id["O"] = 0
                for label in sorted(list(labels)):
                    self.label_to_id[f"B-{label}"] = len(self.label_to_id)
                    self.label_to_id[f"I-{label}"] = len(self.label_to_id)
            else:
                raise ValueError(f"unknown label encoding: {self.label_encoding}")
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

    def encode_input(
        self,
        document: DocumentType,
        is_training: bool = False,
    ) -> List[TaskEncodingType]:
        """
        Create one or multiple task encodings for the given document.
        """
        words = [word.text for word in document.words]
        boxes = [
            normalize_bbox(word.bbox, width=document.image_width, height=document.image_height)
            for word in document.words
        ]
        if document.image_format != "RGB":
            raise NotImplementedError(
                f"document.image_format = {document.image_format} not yet implemented"
            )
        images = torch.tensor(document.image, dtype=torch.uint8)
        encoded_inputs = self.processor(
            images,
            words,
            boxes=boxes,
            # use dummy labels to construct label position mapping later on
            word_labels=list(range(len(words))),
            padding=False,
            truncation=False,
            add_special_tokens=False,
            # max_length=512
        )

        label_positions: List[List[int]] = [[] for _ in range(len(words))]
        for idx, original_label_position in enumerate(encoded_inputs["labels"]):
            if original_label_position != self.label_pad_token_id:
                label_positions[original_label_position].append(idx)

        encoded_inputs.pop("labels")
        image = encoded_inputs.pop("image")[0]

        total_length = len(encoded_inputs["input_ids"])
        chunk_size = self.max_length if self.max_length is not None else total_length
        task_encodings = []
        for start in range(0, total_length, chunk_size):
            end = start + chunk_size
            chunk_data = {k: v[start:end] for k, v in encoded_inputs.items()}
            input_encoding = InputEncoding(
                label_positions=label_positions, image=image, chunk=(start, end), **chunk_data
            )
            task_encodings.append(TaskEncodingType(document=document, inputs=input_encoding))
        return task_encodings

    def encode_target(
        self,
        task_encoding: TaskEncodingType,
    ) -> TargetEncodingType:
        """
        Create a target for a task encoding. This may use any annotations of the underlying document.
        """

        word_mapping = task_encoding.inputs.label_positions
        chunk_start, chunk_end = task_encoding.inputs.chunk
        input_length = len(word_mapping)
        labels = ["O"] * input_length
        for entity in task_encoding.document.entities:
            label = entity.label
            if label in self.exclude_labels:
                continue
            if self.label_encoding == "IOB2":
                labels[entity.start] = f"B-{label}"
                labels[entity.start + 1 : entity.end] = [f"I-{label}"] * (
                    entity.end - entity.start - 1
                )
            else:
                raise ValueError(f"unknown label encoding: {self.label_encoding}")

        word_labels = [self.label_pad_token_id] * (
            len(task_encoding.inputs.input_ids) + chunk_start
        )
        for idx, label in enumerate(labels):
            label_id = self.label_to_id[label]
            target_positions = word_mapping[idx]
            if len(target_positions) == 0:
                continue
            if min(target_positions) >= len(word_labels):
                break
            for pos in target_positions:
                if pos < len(word_labels):
                    word_labels[pos] = label_id

        return TargetEncodingType(word_labels=word_labels[chunk_start:chunk_end])

    def collate(self, task_encodings: Sequence[TaskEncodingType]) -> ModelEncodingType:
        """
        Convert a list of task encodings to a batch that will be passed to the model.
        """

        # get the inputs from the task encodings
        inputs = self.processor.tokenizer.pad(
            [
                {
                    "input_ids": task_encoding.inputs.input_ids,
                    "attention_mask": task_encoding.inputs.attention_mask,
                }
                for task_encoding in task_encodings
            ],
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt",  # if labels is None else None,
        )

        inputs["image"] = torch.tensor(
            [task_encoding.document.image for task_encoding in task_encodings], dtype=torch.uint8
        )

        sequence_length = inputs["input_ids"].shape[1]
        padding_side = self.processor.tokenizer.padding_side

        inputs["bbox"] = pad_sequences(
            [task_encoding.inputs.bbox for task_encoding in task_encodings],
            pad_entry=[0, 0, 0, 0],
            padding_side=padding_side,
            sequence_length=sequence_length,
        )

        inputs = {
            k: torch.tensor(v, dtype=torch.int64) if isinstance(v[0], list) else v
            for k, v in inputs.items()
        }

        if not task_encodings[0].has_targets:
            return inputs, None

        labels_padded = pad_sequences(
            [task_encoding.targets.word_labels for task_encoding in task_encodings],
            pad_entry=self.label_pad_token_id,
            padding_side=padding_side,
            sequence_length=sequence_length,
        )
        targets = torch.tensor(labels_padded, dtype=torch.int64)

        return inputs, targets

    def unbatch_output(self, model_output: ModelOutputType) -> Sequence[TaskOutputType]:
        """
        Convert one model output batch to a sequence of taskmodule outputs.
        """
        logits = model_output.logits
        batch_size, sequence_length, _ = logits.shape
        probabilities = softmax(logits, dim=-1).detach().cpu().numpy()
        indices = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        tags = [[self.id_to_label[e] for e in b] for b in indices]
        return [
            TaskOutput(tags=t, probabilities=probabilities[batch_idx].take(indices[batch_idx]))
            for batch_idx, t in enumerate(tags)
        ]

    def create_annotations_from_output(
        self,
        task_encodings: TaskEncodingType,
        task_outputs: TaskOutputType,
    ) -> Iterator[Tuple[str, Label]]:
        """
        Convert a task output to annotations. The method has to yield tuples (annotation_name, annotation).
        """

        if self.label_encoding == "IOB2":
            none_tag = "O"
            tags_to_spans_function = bio_tags_to_spans
        else:
            raise ValueError(f"unknown label encoding: {self.label_encoding}")
        tag_sequence = []
        probabilities = []
        chunk_start, chunk_end = task_encodings.inputs.chunk
        for original_idx, output_indices in enumerate(task_encodings.inputs.label_positions):
            if len(output_indices) > 0:
                # output_indices can contain multiple entries if:
                # 1. self.processor.tokenizer.only_label_first_subword==False, or
                # 2. a word starts with a subword token that marks space, i.e. "▁"
                if chunk_start <= min(output_indices) and max(output_indices) < chunk_end:
                    # For now, we simply use the first entry
                    output_idx = output_indices[0]
                    tag_sequence.append(task_outputs.tags[output_idx - chunk_start])
                    probabilities.append(task_outputs.probabilities[output_idx - chunk_start])
                else:
                    tag_sequence.append(none_tag)
                    probabilities.append(None)
            else:
                tag_sequence.append(none_tag)
                probabilities.append(None)

        spans = tags_to_spans_function(
            tag_sequence, include_ill_formed=self.include_ill_formed_predictions
        )

        for label, (start, end_inclusive) in spans:
            end = end_inclusive + 1
            valid_probs = [prob for prob in probabilities[start:end] if prob is not None]
            if len(valid_probs) > 0:
                # take average probability as score
                score = sum(valid_probs) / len(valid_probs)
                yield "entities", OcrLabeledSpan(start=start, end=end, label=label, score=score)
