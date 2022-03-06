from typing import (
    Any,
    Dict,
    Iterator,
    List,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    Union,
)

import numpy as np
import torch
from transformers import AutoTokenizer
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy

from pytorch_ie.data.document import Annotation, Document, Label
from pytorch_ie.models.transformer_text_classification import (
    TransformerTextClassificationModelBatchOutput,
    TransformerTextClassificationModelStepBatchEncoding,
)
from pytorch_ie.taskmodules.taskmodule import Metadata, TaskEncoding, TaskModule

"""
workflow:
    Document
        -> (InputEncoding, TargetEncoding) -> TaskEncoding -> TaskBatchEncoding
            -> ModelBatchEncoding -> ModelBatchOutput
        -> TaskOutput
    -> Document
"""

TransformerTextClassificationInputEncoding = MutableMapping[str, Any]
TransformerTextClassificationTargetEncoding = List[int]

TransformerTextClassificationTaskEncoding = TaskEncoding[
    TransformerTextClassificationInputEncoding, TransformerTextClassificationTargetEncoding
]


class TransformerTextClassificationTaskOutputSingle(TypedDict, total=False):
    labels: List[str]
    probabilities: List[float]


class TransformerTextClassificationTaskOutputMulti(TypedDict, total=False):
    labels: List[List[str]]
    probabilities: List[List[float]]


TransformerTextClassificationTaskOutput = Union[
    TransformerTextClassificationTaskOutputSingle,
    TransformerTextClassificationTaskOutputMulti,
]

_TransformerTextClassificationTaskModule = TaskModule[
    # _InputEncoding, _TargetEncoding, _TaskBatchEncoding, _ModelBatchOutput, _TaskOutput
    TransformerTextClassificationInputEncoding,
    TransformerTextClassificationTargetEncoding,
    TransformerTextClassificationModelStepBatchEncoding,
    TransformerTextClassificationModelBatchOutput,
    TransformerTextClassificationTaskOutput,
]


class TransformerTextClassificationTaskModule(_TransformerTextClassificationTaskModule):
    def __init__(
        self,
        tokenizer_name_or_path: str,
        label_to_verbalizer: Dict[str, str],
        annotation: str = "labels",
        padding: Union[bool, str, PaddingStrategy] = True,
        truncation: Union[bool, str, TruncationStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        multi_label: bool = False,
        label_to_id: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

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

    def _config(self) -> Dict[str, Any]:
        config = super()._config()
        config["label_to_id"] = self.label_to_id
        return config

    def prepare(self, documents: List[Document]) -> None:
        labels = set()
        for document in documents:
            annotations = document.annotations.labels[self.annotation]

            for annotation in annotations:
                # TODO: labels is a set...
                for label in annotation.labels:
                    if label not in labels:
                        labels.add(label)

        self.label_to_id["O"] = 0
        current_id = 1
        for label in labels:
            self.label_to_id[label] = current_id
            current_id += 1

        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

    def encode_input(
        self,
        documents: List[Document],
        is_training: bool = False,
    ) -> Tuple[
        List[TransformerTextClassificationInputEncoding],
        List[Metadata],
        Optional[List[Document]],
    ]:
        input_encoding = [
            self.tokenizer(
                doc.text,
                padding=False,
                truncation=self.truncation,
                max_length=self.max_length,
                is_split_into_words=False,
                return_offsets_mapping=True,
                return_special_tokens_mask=True,
            )
            for doc in documents
        ]

        metadata = [
            {
                "offset_mapping": encoding.pop("offset_mapping"),
                "special_tokens_mask": encoding.pop("special_tokens_mask"),
            }
            for encoding in input_encoding
        ]

        return input_encoding, metadata, documents

    def encode_target(
        self,
        documents: List[Document],
        input_encodings: List[TransformerTextClassificationInputEncoding],
        metadata: List[Metadata],
    ) -> List[TransformerTextClassificationTargetEncoding]:

        target: List[TransformerTextClassificationTargetEncoding] = []
        for i, document in enumerate(documents):
            annotations = document.annotations.labels[self.annotation]
            if self.multi_label:
                label_ids = [0] * len(self.label_to_id)
                for annotation in annotations:
                    for label in annotation.labels:
                        label_id = self.label_to_id[label]
                        label_ids[label_id] = 1
            else:
                assert len(annotations) == 1 and not annotations[0].is_multilabel

                label = annotations[0].label_single
                label_ids = [self.label_to_id[label]]

            target.append(label_ids)

        return target

    def unbatch_output(
        self, output: TransformerTextClassificationModelBatchOutput
    ) -> Sequence[TransformerTextClassificationTaskOutput]:
        logits = output["logits"]

        output_label_probs = logits.sigmoid() if self.multi_label else logits.softmax(dim=-1)
        output_label_probs = output_label_probs.detach().cpu().numpy()

        if self.multi_label:
            raise NotImplementedError
            # labels = [[] for _ in range(batch_size)]
            # probabilities = [[] for _ in range(batch_size)]
            # for batch_idx in range(batch_size):
            #     for label_idx in range(num_labels):
            #         prob = label_probs[batch_idx, label_idx]
            #         if prob > 0.5:
            #             label = index_to_label[label_idx]
            #             labels[batch_idx].append(label)
            #             probabilities[batch_idx].append(prob)

            # labels = [[self.id_to_label[e] for e in b] for b in label_ids]
            # labels = []
            # for instance_label_probs in output_label_probs:
            #     instance_labels = []
            #     for label_id in example_label_ids:
            #         example_labels.append(self.id_to_label[label_id])

        else:
            decoded_output = []
            label_ids = np.argmax(output_label_probs, axis=-1)
            for batch_idx, label_id in enumerate(label_ids):
                label = self.id_to_label[label_id]
                prob = float(output_label_probs[batch_idx, label_id])
                result: TransformerTextClassificationTaskOutputSingle = {
                    "labels": [label],
                    "probabilities": [prob],
                }

                decoded_output.append(result)

            return decoded_output

    def create_annotations_from_output(
        self,
        encoding: TransformerTextClassificationTaskEncoding,
        output: TransformerTextClassificationTaskOutput,
    ) -> Iterator[Tuple[str, Annotation]]:
        if self.multi_label:
            # Note: we can not use isinstance since that does not work with TypedDicts
            multi_output: TransformerTextClassificationTaskOutputMulti = output  # type: ignore
            for labels, probabilities in zip(
                multi_output["labels"], multi_output["probabilities"]
            ):
                yield self.annotation, Label(label=labels[0], score=probabilities[0])
        else:
            # Note: we can not use isinstance since that does not work with TypedDicts
            single_output: TransformerTextClassificationTaskOutputSingle = output  # type: ignore
            for label, probability in zip(single_output["labels"], single_output["probabilities"]):
                yield self.annotation, Label(label=label, score=probability)

    def collate(
        self, encodings: List[TransformerTextClassificationTaskEncoding]
    ) -> TransformerTextClassificationModelStepBatchEncoding:
        input_features = [encoding.input for encoding in encodings]
        metadata = [encoding.metadata for encoding in encodings]
        documents = [encoding.document for encoding in encodings]

        input_ = self.tokenizer.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if not encodings[0].has_target:
            return input_, None

        target_list: List[TransformerTextClassificationTargetEncoding] = [
            encoding.target for encoding in encodings
        ]

        target = torch.tensor(target_list, dtype=torch.int64)

        if not self.multi_label:
            target = target.flatten()

        return input_, target
