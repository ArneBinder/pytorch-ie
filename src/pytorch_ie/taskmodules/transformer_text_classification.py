"""
workflow:
    document
        -> (input_encoding, target_encoding) -> task_encoding
            -> model_encoding -> model_output
        -> task_output
    -> document
"""
import logging
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypedDict,
    Union,
)

import numpy as np
import torch
from transformers import AutoTokenizer
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy
from typing_extensions import TypeAlias

from pytorch_ie.annotations import Label, MultiLabel
from pytorch_ie.core import TaskEncoding, TaskModule
from pytorch_ie.documents import TextDocument, TextDocumentWithLabel, TextDocumentWithMultiLabel
from pytorch_ie.models.transformer_text_classification import ModelOutputType, ModelStepInputType

logger = logging.getLogger(__name__)

InputEncodingType: TypeAlias = MutableMapping[str, Any]
TargetEncodingType: TypeAlias = Sequence[int]

TaskEncodingType: TypeAlias = TaskEncoding[
    TextDocument,
    InputEncodingType,
    TargetEncodingType,
]


class TaskOutputSingleType(TypedDict, total=False):
    labels: Sequence[str]
    probabilities: Sequence[float]


class TaskOutputMultiType(TypedDict, total=False):
    labels: Sequence[Sequence[str]]
    probabilities: Sequence[Sequence[float]]


TaskOutputType: TypeAlias = Union[
    TaskOutputSingleType,
    TaskOutputMultiType,
]

TaskModuleType: TypeAlias = TaskModule[
    TextDocument,
    InputEncodingType,
    TargetEncodingType,
    ModelStepInputType,
    ModelOutputType,
    TaskOutputType,
]


@TaskModule.register()
class TransformerTextClassificationTaskModule(TaskModuleType):
    def __init__(
        self,
        tokenizer_name_or_path: str,
        label_to_verbalizer: Dict[str, str],
        annotation: str = "label",
        padding: Union[bool, str, PaddingStrategy] = True,
        truncation: Union[bool, str, TruncationStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        multi_label: bool = False,
        label_to_id: Optional[Dict[str, int]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters()

        if multi_label:
            raise NotImplementedError(
                "Multi-label classification (multi_label=True) is not supported yet."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

        self.annotation = annotation
        self.label_to_verbalizer = label_to_verbalizer
        self.padding = padding
        self.truncation = truncation
        self.label_to_id = label_to_id or {}
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.multi_label = multi_label

    @property
    def document_type(self) -> Optional[Type[TextDocument]]:
        dt: Type[TextDocument]
        if self.multi_label:
            dt = TextDocumentWithMultiLabel
        else:
            dt = TextDocumentWithLabel
        if self.annotation == "label":
            return dt
        else:
            logger.warning(
                f"annotation={self.annotation} is "
                f"not the default value ('label'), so the taskmodule {type(self).__name__} can not request "
                f"the usual document type ({dt.__name__}) for auto-conversion because this has the bespoken "
                f"default value as layer name instead of the provided one."
            )
            return None

    def _config(self) -> Dict[str, Any]:
        config = super()._config()
        config["label_to_id"] = self.label_to_id
        return config

    def prepare(self, documents: Sequence[TextDocument]) -> None:
        labels = set()
        for document in documents:
            annotations: Sequence[Label] = document[self.annotation]

            for annotation in annotations:
                labels.add(annotation.label)

        self.label_to_id["O"] = 0
        current_id = 1
        for label in sorted(labels):
            self.label_to_id[label] = current_id
            current_id += 1

        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

    def encode_input(
        self,
        document: TextDocument,
        is_training: bool = False,
    ) -> Optional[Union[TaskEncodingType, Sequence[TaskEncodingType],]]:
        inputs = self.tokenizer(
            document.text,
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

        return TaskEncoding(
            document=document,
            inputs=inputs,
            metadata=metadata,
        )

    def encode_target(
        self,
        task_encoding: TaskEncodingType,
    ) -> TargetEncodingType:
        label_annotation: Sequence[Union[Label, MultiLabel]] = task_encoding.document[
            self.annotation
        ]

        targets: TargetEncodingType
        if self.multi_label:
            assert isinstance(label_annotation, MultiLabel)
            targets = [0] * len(self.label_to_id)
            for label in label_annotation.label:
                label_id = self.label_to_id[label]
                targets[label_id] = 1
        else:
            assert isinstance(label_annotation, Label)
            label = label_annotation.label
            targets = [self.label_to_id[label]]

        return targets

    def unbatch_output(self, model_output: ModelOutputType) -> Sequence[TaskOutputType]:
        logits = model_output["logits"]

        output_label_probs = logits.sigmoid() if self.multi_label else logits.softmax(dim=-1)
        output_label_probs = output_label_probs.detach().cpu().numpy()

        if self.multi_label:
            raise NotImplementedError()

        else:
            unbatched_output = []
            label_ids = np.argmax(output_label_probs, axis=-1)
            for batch_idx, label_id in enumerate(label_ids):
                label = self.id_to_label[label_id]
                prob = float(output_label_probs[batch_idx, label_id])
                result: TaskOutputSingleType = {
                    "labels": [label],
                    "probabilities": [prob],
                }

                unbatched_output.append(result)

            return unbatched_output

    def create_annotations_from_output(
        self,
        task_encodings: TaskEncodingType,
        task_outputs: TaskOutputType,
    ) -> Iterator[Tuple[str, Union[Label, MultiLabel]]]:
        if self.multi_label:
            # Note: we can not use isinstance since that does not work with TypedDicts
            multi_output: TaskOutputMultiType = task_outputs  # type: ignore
            for labels, probabilities in zip(
                multi_output["labels"], multi_output["probabilities"]
            ):
                yield self.annotation, Label(label=labels[0], score=probabilities[0])
        else:
            # Note: we can not use isinstance since that does not work with TypedDicts
            single_output: TaskOutputSingleType = task_outputs  # type: ignore
            for label, probability in zip(single_output["labels"], single_output["probabilities"]):
                yield self.annotation, Label(label=label, score=probability)

    def collate(self, task_encodings: Sequence[TaskEncodingType]) -> ModelStepInputType:
        input_features = [task_encoding.inputs for task_encoding in task_encodings]
        metadata = [task_encoding.metadata for task_encoding in task_encodings]
        documents = [task_encoding.document for task_encoding in task_encodings]

        inputs = self.tokenizer.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if not task_encodings[0].has_targets:
            return inputs, None

        target_list: List[TargetEncodingType] = [
            task_encoding.targets for task_encoding in task_encodings
        ]

        targets = torch.tensor(target_list, dtype=torch.int64)

        if not self.multi_label:
            targets = targets.flatten()

        return inputs, targets
