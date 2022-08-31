from typing import Any, Dict, Iterator, MutableMapping, Optional, Sequence, Tuple, TypedDict, Union

import numpy as np
import torch
from transformers import AutoTokenizer
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy

from pytorch_ie.annotations import Label
from pytorch_ie.core import TaskEncoding, TaskModule
from pytorch_ie.documents import TextDocument
from pytorch_ie.models.transformer_text_classification import (
    TransformerTextClassificationModelBatchOutput,
    TransformerTextClassificationModelStepBatchEncoding,
)

"""
workflow:
    Document
        -> TaskEncoding(InputEncoding, TargetEncoding) -> TaskBatchEncoding
            -> ModelBatchEncoding -> ModelBatchOutput
        -> TaskOutput
    -> Document
"""

TransformerTextClassificationInputEncoding = MutableMapping[str, Any]
TransformerTextClassificationTargetEncoding = int

TransformerTextClassificationTaskEncoding = TaskEncoding[
    TextDocument,
    TransformerTextClassificationInputEncoding,
    TransformerTextClassificationTargetEncoding,
]


class TransformerTextClassificationTaskOutput(TypedDict, total=False):
    label: str
    probability: float


_TransformerSimpleTextClassificationTaskModule = TaskModule[
    # _InputEncoding, _TargetEncoding, _TaskBatchEncoding, _ModelBatchOutput, _TaskOutput
    TextDocument,
    TransformerTextClassificationInputEncoding,
    TransformerTextClassificationTargetEncoding,
    TransformerTextClassificationModelStepBatchEncoding,
    TransformerTextClassificationModelBatchOutput,
    TransformerTextClassificationTaskOutput,
]


@TaskModule.register()
class TransformerSimpleTextClassificationTaskModule(_TransformerSimpleTextClassificationTaskModule):
    def __init__(
        self,
        tokenizer_name_or_path: str,
        annotation: str = "labels",
        padding: Union[bool, str, PaddingStrategy] = True,
        truncation: Union[bool, str, TruncationStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        label_to_id: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__()
        # Save all passed arguments. They will be available via self._config().
        self.save_hyperparameters()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

        # this is the name of the annotation of the document that wil be classified
        self.annotation = annotation

        # some tokenization parameters
        self.padding = padding
        self.truncation = truncation
        self.max_length = max_length
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

    def prepare(self, documents: Sequence[TextDocument]) -> None:
        """
        Prepare the task module with training documents, e.g. collect all possible labels.
        """

        # create the label-to-id mapping
        labels = set()
        for document in documents:
            # all annotations of a document are hold in list like containers,
            # so we have to take its first element
            label = document[self.annotation][0]
            labels.add(label)

        # create the mapping, but spare the first index for the "O" (outside) class
        self.label_to_id = {label: i + 1 for i, label in enumerate(sorted(labels))}
        self.label_to_id["O"] = 0
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

    def encode_input(
        self,
        document: TextDocument,
        is_training: bool = False,
    ) -> TransformerTextClassificationTaskEncoding:
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
        task_encoding: TransformerTextClassificationTaskEncoding,
    ) -> TransformerTextClassificationTargetEncoding:
        """
        Create a target for a task encoding. This may use any annotations of the underlying document.
        """

        # as above, all annotations are hold in lists, so we have to take its first element
        label_annotation: Label = task_encoding.document[self.annotation][0]
        # translate the textual label to the target id
        return self.label_to_id[label_annotation.label]

    def unbatch_output(
        self, model_output: TransformerTextClassificationModelBatchOutput
    ) -> Sequence[TransformerTextClassificationTaskOutput]:
        """
        Convert one model output batch to a sequence of taskmodule outputs.
        """

        # get the logits from the model output
        logits = model_output["logits"]

        # normalize the logits to get "probabilities"
        probabilities = logits.softmax(dim=-1).detach().cpu().numpy()

        # get the max class index per example
        max_label_ids = np.argmax(probabilities, axis=-1)

        unbatched_output = []
        for idx, label_id in enumerate(max_label_ids):
            # translate the label id back to the label text
            label = self.id_to_label[label_id]
            # get the probability and convert from tensor value to python float
            prob = float(probabilities[idx, label_id])
            # we create TransformerTextClassificationTaskOutput primarily for typing purposes,
            # a simple dict would also work
            result: TransformerTextClassificationTaskOutput = {
                "label": label,
                "probability": prob,
            }
            unbatched_output.append(result)

            return unbatched_output

    def create_annotations_from_output(
        self,
        task_encodings: TransformerTextClassificationTaskEncoding,
        task_outputs: TransformerTextClassificationTaskOutput,
    ) -> Iterator[Tuple[str, Label]]:
        """
        Convert a task output to annotations. The method has to yield tuples (annotation_name, annotation).
        """

        # just yield a single annotation (other tasks may need multiple annotations per task output)
        yield self.annotation, Label(
            label=task_outputs["label"], score=task_outputs["probability"]
        )

    def collate(
        self, task_encodings: Sequence[TransformerTextClassificationTaskEncoding]
    ) -> TransformerTextClassificationModelStepBatchEncoding:
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
