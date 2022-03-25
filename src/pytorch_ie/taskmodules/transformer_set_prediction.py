from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, TypedDict, Union

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import BatchEncoding, TruncationStrategy

from pytorch_ie.data import Metadata
from pytorch_ie.data.document import Annotation, Document, LabeledSpan
from pytorch_ie.models.transformer_set_prediction import (
    TransformerSetPredictionModelBatchOutput,
    TransformerSetPredictionModelStepBatchEncoding,
)
from pytorch_ie.taskmodules.taskmodule import TaskEncoding, TaskModule

TransformerSetPredictionInputEncoding = BatchEncoding
# example of TransformerSetPredictionTargetEncoding:
# {'entities': {'start_index': [7, 1, 3], 'end_index': [7, 1, 3], 'label_ids': [0, 3, 0], 'span_mask': [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]}}
TransformerSetPredictionTargetEncoding = Dict[str, Dict[str, Any]]

TransformerSetPredictionTaskEncoding = TaskEncoding[
    TransformerSetPredictionInputEncoding, TransformerSetPredictionTargetEncoding
]


class TransformerSetPredictionTaskOutput(TypedDict, total=False):
    tags: List[Tuple[str, Tuple[int, int]]]
    probabilities: List[float]


_TransformerSetPredictionTaskModule = TaskModule[
    # _InputEncoding, _TargetEncoding, _TaskBatchEncoding, _ModelBatchOutput, _TaskOutput
    TransformerSetPredictionInputEncoding,
    TransformerSetPredictionTargetEncoding,
    TransformerSetPredictionModelStepBatchEncoding,
    TransformerSetPredictionModelBatchOutput,
    TransformerSetPredictionTaskOutput,
]


class TransformerSetPredictionTaskModule(_TransformerSetPredictionTaskModule):
    def __init__(
        self,
        tokenizer_name_or_path: str,
        entity_annotation: str = "entities",
        partition_annotation: Optional[str] = None,
        padding: Union[bool, str, PaddingStrategy] = True,
        truncation: Union[bool, str, TruncationStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        label_pad_token_id: int = -100,
        label_to_id: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.entity_annotation = entity_annotation
        self.partition_annotation = partition_annotation
        self.label_to_id = label_to_id or {}
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        self.padding = padding
        self.truncation = truncation
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id

    def _config(self) -> Dict[str, Any]:
        config = super()._config()
        config["label_to_id"] = self.label_to_id
        return config

    def prepare(self, documents: List[Document]) -> None:
        labels = set()
        for document in documents:
            entities = document.annotations.spans[self.entity_annotation]

            for entity in entities:
                entity_labels = entity.label if entity.is_multilabel else [entity.label]
                for label in entity_labels:
                    if label not in labels:
                        labels.add(label)

        # self.label_to_id["O"] = 0
        # current_id = 1
        current_id = 0
        for label in labels:
            self.label_to_id[label] = current_id
            current_id += 1

        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

    def encode_input(
        self, documents: List[Document]
    ) -> Tuple[
        List[TransformerSetPredictionInputEncoding], List[Metadata], Optional[List[Document]]
    ]:
        # TODO: simplify (see other taskmodules)
        expanded_documents = None
        if self.partition_annotation is not None:
            input_ = [
                self.tokenizer(
                    doc.text[sent.start : sent.end],
                    padding=False,
                    truncation=self.truncation,
                    max_length=self.max_length,
                    is_split_into_words=False,
                    return_offsets_mapping=True,
                    return_special_tokens_mask=True,
                )
                for doc in documents
                for sent in doc.annotations.spans[self.partition_annotation]
            ]
            expanded_documents = [
                doc for doc in documents for _ in doc.annotations.spans[self.partition_annotation]
            ]
        else:
            input_ = [
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
                "offset_mapping": inp.pop("offset_mapping"),
                "special_tokens_mask": inp.pop("special_tokens_mask"),
            }
            for inp in input_
        ]

        if self.partition_annotation is not None:
            i = 0
            for document in documents:
                for sentence_index in range(
                    len(document.annotations.spans[self.partition_annotation])
                ):
                    metadata[i]["sentence_index"] = sentence_index
                    i += 1

        return input_, metadata, expanded_documents

    def encode_target(
        self,
        documents: List[Document],
        input_: List[TransformerSetPredictionInputEncoding],
        metadata: List[Metadata],
    ) -> List[TransformerSetPredictionTargetEncoding]:
        target = []
        i = 0
        for document in documents:
            entities = document.annotations.spans[self.entity_annotation]

            if self.partition_annotation is not None:
                partitions = document.annotations.spans[self.partition_annotation]
            else:
                partitions = [LabeledSpan(start=0, end=len(document.text), label="FULL_DOCUMENT")]

            for partition in partitions:
                start_indices = []
                end_indices = []
                label_ids = []
                span_masks = []
                for entity in entities:
                    if entity.start < partition.start or entity.end > partition.end:
                        continue

                    entity_start = entity.start - partition.start
                    entity_end = entity.end - partition.start

                    start_idx = input_[i].char_to_token(entity_start)
                    end_idx = input_[i].char_to_token(entity_end - 1)

                    if start_idx is None or end_idx is None:
                        continue

                    start_indices.append(start_idx)
                    end_indices.append(end_idx)
                    label_ids.append(self.label_to_id[entity.label])

                    span_mask = [
                        1 if start_idx <= i <= end_idx else 0
                        for i in range(len(input_[i].word_ids()))
                    ]
                    span_masks.append(span_mask)

                target.append(
                    {
                        "entities": {
                            "start_index": start_indices,
                            "end_index": end_indices,
                            "label_ids": label_ids,
                            # "span_position": span_positions,
                            "span_mask": span_masks,
                        }
                    }
                )
                i += 1
        return target

    def unbatch_output(
        self, output: TransformerSetPredictionModelBatchOutput
    ) -> Sequence[TransformerSetPredictionTaskOutput]:
        predictions = output["entities"]

        label_ids_pred_full = (
            F.softmax(predictions["label_ids"], dim=-1).argmax(dim=-1).cpu().numpy()
        )
        start_index_pred_full = (
            F.softmax(predictions["start_index"], dim=-1).argmax(dim=-1).cpu().numpy()
        )
        end_index_pred_full = (
            F.softmax(predictions["end_index"], dim=-1).argmax(dim=-1).cpu().numpy()
        )

        tags = []
        probabilities = []
        for label_ids_pred, start_index_pred, end_index_pred in zip(
            label_ids_pred_full, start_index_pred_full, end_index_pred_full
        ):
            indices_pred = label_ids_pred != len(self.label_to_id)

            label_ids = label_ids_pred[indices_pred]
            start_indices = start_index_pred[indices_pred]
            end_indices = end_index_pred[indices_pred]

            current_tags = []
            current_probabilities = []
            for label_id, start, end in zip(label_ids, start_indices, end_indices):
                label = self.id_to_label[label_id]
                current_tags.append((label, (start, end)))
                current_probabilities.append(1.0)

            tags.append(current_tags)
            probabilities.append(current_probabilities)

        # labels = [[self.id_to_label[e] for e in b] for b in label_ids]
        return [
            TransformerSetPredictionTaskOutput(tags=t, probabilities=p)
            for t, p in zip(tags, probabilities)
        ]

    def create_annotations_from_output(
        self,
        encoding: TransformerSetPredictionTaskEncoding,
        output: TransformerSetPredictionTaskOutput,
    ) -> Iterator[Tuple[str, Annotation]]:

        document = encoding.document
        metadata = encoding.metadata

        if self.partition_annotation is not None:
            sentence = document.annotations.spans[self.partition_annotation][
                metadata["sentence_index"]
            ]
            offset = sentence.start
        else:
            offset = 0

        # tag_sequence = [
        #     "O" if stm else tag
        #     for tag, stm in zip(output["tags"], metadata["special_tokens_mask"])
        # ]

        # spans = bio_tags_to_spans(tag_sequence)
        spans = output["tags"]
        for label, (start, end) in spans:
            yield self.entity_annotation, LabeledSpan(
                offset + metadata["offset_mapping"][start][0],
                offset + metadata["offset_mapping"][end][1],
                label,
            )

    def collate(
        self, encodings: List[TransformerSetPredictionTaskEncoding]
    ) -> TransformerSetPredictionModelStepBatchEncoding:
        input_features = [encoding.input for encoding in encodings]

        input_ = self.tokenizer.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if not encodings[0].has_target:
            return input_, None

        target: Dict[str, Dict[str, List[torch.Tensor]]] = {}
        for encoding in encodings:
            for decoder, decoder_target in encoding.target.items():
                if decoder not in target:
                    target[decoder] = {}
                for key, val in decoder_target.items():
                    if key not in target[decoder]:
                        target[decoder][key] = []
                    if val and isinstance(val[0], list):
                        if key.endswith("_mask"):
                            seq_length = input_["input_ids"].shape[1]
                            val_tensor = []
                            for v in val:
                                padding_length = seq_length - len(v)
                                val_tensor.append(v + padding_length * [-100])

                            target[decoder][key].append(
                                torch.tensor(val_tensor, dtype=torch.float)
                            )
                        else:
                            target[decoder][key].append(
                                [torch.tensor(v, dtype=torch.float) for v in val]
                            )
                    else:
                        target[decoder][key].append(torch.tensor(val, dtype=torch.float))

        input_ = {k: torch.tensor(v, dtype=torch.int64) for k, v in input_.items()}

        return input_, target
