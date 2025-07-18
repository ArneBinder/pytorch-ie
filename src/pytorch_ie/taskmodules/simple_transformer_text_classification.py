"""
workflow:
    document
        -> (input_encoding, target_encoding) -> task_encoding
            -> model_encoding -> model_output
        -> task_output
    -> document
"""

import logging
from typing import Any, Dict, Iterator, MutableMapping, Optional, Sequence, Tuple, TypedDict, Union

import numpy as np
import torch
from pie_core import TaskEncoding, TaskModule
from transformers import AutoTokenizer
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy
from typing_extensions import TypeAlias

from pytorch_ie.annotations import Label
from pytorch_ie.documents import TextDocumentWithLabel
from pytorch_ie.models.transformer_text_classification import ModelOutputType, ModelStepInputType

logger = logging.getLogger(__name__)


class TaskOutput(TypedDict, total=False):
    label: str
    probability: float


# Define task specific input and output types
DocumentType: TypeAlias = TextDocumentWithLabel
InputEncodingType: TypeAlias = MutableMapping[str, Any]
TargetEncodingType: TypeAlias = int
ModelEncodingType: TypeAlias = ModelStepInputType
ModelOutputType: TypeAlias = ModelOutputType
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


@TaskModule.register()
class SimpleTransformerTextClassificationTaskModule(TaskModuleType):
    # If these attributes are set, the taskmodule is considered as prepared. They should be calculated
    # within _prepare() and are dumped automatically when saving the taskmodule with save_pretrained().
    PREPARED_ATTRIBUTES = ["label_to_id"]
    DOCUMENT_TYPE = TextDocumentWithLabel

    def __init__(
        self,
        tokenizer_name_or_path: str,
        padding: Union[bool, str, PaddingStrategy] = True,
        truncation: Union[bool, str, TruncationStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        label_to_id: Optional[Dict[str, int]] = None,
        **kwargs,
    ) -> None:
        # Important: Remaining keyword arguments need to be passed to super.
        super().__init__(**kwargs)
        # Save all passed arguments. They will be available via self._config().
        self.save_hyperparameters()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

        # some tokenization and padding parameters
        self.truncation = truncation
        self.max_length = max_length
        self.padding = padding
        self.pad_to_multiple_of = pad_to_multiple_of

        # this will be prepared from the data or loaded from the config
        self.label_to_id = label_to_id

    def _prepare(self, documents: Sequence[DocumentType]) -> None:
        """
        Prepare the task module with training documents, e.g. collect all possible labels.
        This method needs to set all attributes listed in PREPARED_ATTRIBUTES.
        """

        # create the label-to-id mapping
        labels = set()
        for document in documents:
            # all annotations of a document are hold in list like containers,
            # so we have to take its first element
            label_annotation = document.label[0]
            labels.add(label_annotation.label)

        # create the mapping, but spare the first index for the "O" (outside) class
        self.label_to_id = {label: i + 1 for i, label in enumerate(sorted(labels))}
        self.label_to_id["O"] = 0

    def _post_prepare(self):
        """
        Any further preparation logic that requires the result of _prepare(). But its result is not serialized
        with the taskmodule.
        """

        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

    def encode_input(
        self,
        document: DocumentType,
    ) -> TaskEncodingType:
        """
        Create one or multiple task encodings for the given document.
        """

        # tokenize the input text, this will be the input
        inputs = self.tokenizer(
            document.text,
            # we do not pad here, this will be done in collate()
            # when the actual batches are created
            padding=False,
            truncation=self.truncation,
            max_length=self.max_length,
        )

        return TaskEncoding(
            document=document,
            inputs=inputs,
        )

    def encode_target(
        self,
        task_encoding: TaskEncodingType,
    ) -> TargetEncodingType:
        """
        Create a target for a task encoding. This may use any annotations of the underlying document.
        """

        # as above, all annotations are hold in lists, so we have to take its first element
        label_annotation = task_encoding.document.label[0]
        # translate the textual label to the target id
        assert self.label_to_id is not None
        return self.label_to_id[label_annotation.label]

    def collate(self, task_encodings: Sequence[TaskEncodingType]) -> ModelEncodingType:
        """
        Convert a list of task encodings to a batch that will be passed to the model.
        """
        # get the inputs from the task encodings
        input_features = [task_encoding.inputs for task_encoding in task_encodings]

        # pad the inputs and return torch tensors
        inputs = self.tokenizer.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if task_encodings[0].has_targets:
            # convert the targets (label ids) to a tensor
            targets = torch.tensor(
                [task_encoding.targets for task_encoding in task_encodings], dtype=torch.int64
            )
        else:
            # during inference, we do not have any targets
            targets = None

        return inputs, targets

    def unbatch_output(self, model_output: ModelOutputType) -> Sequence[TaskOutputType]:
        """
        Convert one model output batch to a sequence of taskmodule outputs.
        """

        # get the logits from the model output
        logits = model_output["logits"]

        # convert the logits to "probabilities"
        probabilities = logits.softmax(dim=-1).detach().cpu().float().numpy()

        # get the max class index per example
        max_label_ids = np.argmax(probabilities, axis=-1)

        outputs = []
        for idx, label_id in enumerate(max_label_ids):
            # translate the label id back to the label text
            label = self.id_to_label[label_id]
            # get the probability and convert from tensor value to python float
            prob = float(probabilities[idx, label_id])
            # we create TransformerTextClassificationTaskOutput primarily for typing purposes,
            # a simple dict would also work
            result: TaskOutput = {
                "label": label,
                "probability": prob,
            }
            outputs.append(result)

        return outputs

    def create_annotations_from_output(
        self,
        task_encodings: TaskEncodingType,
        task_outputs: TaskOutputType,
    ) -> Iterator[Tuple[str, Label]]:
        """
        Convert a task output to annotations. The method has to yield tuples (annotation_name, annotation).
        """

        # just yield a single annotation (other tasks may need multiple annotations per task output)
        yield "label", Label(label=task_outputs["label"], score=task_outputs["probability"])
