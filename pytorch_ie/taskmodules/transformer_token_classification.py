import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy

from pytorch_ie.data.document import Document, LabeledSpan, Annotation
from pytorch_ie.data.span_utils import bio_tags_to_spans
from pytorch_ie.taskmodules.taskmodule import TaskEncoding, TaskModule


class TransformerTokenClassificationTaskModule(TaskModule):
    def __init__(
        self,
        tokenizer_name_or_path: str,
        entity_annotation: str = "entities",
        single_sentence: bool = False,
        sentence_annotation: str = "sentences",
        padding: Union[bool, str, PaddingStrategy] = True,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        label_pad_token_id: int = -100,
        label_to_id: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__(
            tokenizer_name_or_path=tokenizer_name_or_path,
            entity_annotation=entity_annotation,
            single_sentence=single_sentence,
            sentence_annotation=sentence_annotation,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            label_pad_token_id=label_pad_token_id,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.entity_annotation = entity_annotation
        self.single_sentence = single_sentence
        self.sentence_annotation = sentence_annotation
        self.label_to_id = label_to_id or {}
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        self.padding = padding
        self.truncation = truncation
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id

    def _config(self) -> Optional[Dict[str, Any]]:
        config = super()._config()
        config["label_to_id"] = self.label_to_id
        return config

    def prepare(self, documents: List[Document]) -> None:
        labels = set()
        for document in documents:
            entities = document.annotations(self.entity_annotation)

            for entity in entities:
                entity_labels = entity.label if entity.is_multilabel else [entity.label]
                for label in entity_labels:
                    if label not in labels:
                        labels.add(label)

        self.label_to_id["O"] = 0
        current_id = 1
        for label in labels:
            for prefix in ["B", "I"]:
                self.label_to_id[f"{prefix}-{label}"] = current_id
                current_id += 1

        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

    def encode_input(
        self, documents: List[Document]
    ) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]], Optional[List[Document]]]:
        expanded_documents = None
        if self.single_sentence:
            input_ = [
                self.tokenizer(
                    doc.text[sent.start : sent.end],
                    padding=False,
                    truncation=False,
                    max_length=None,
                    is_split_into_words=False,
                    return_offsets_mapping=True,
                    return_special_tokens_mask=True,
                )
                for doc in documents
                for sent in doc.annotations(self.sentence_annotation)
            ]
            expanded_documents = [
                doc for doc in documents for _ in doc.annotations(self.sentence_annotation)
            ]
        else:
            input_ = [
                self.tokenizer(
                    doc.text,
                    padding=False,
                    truncation=False,
                    max_length=None,
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

        if self.single_sentence:
            i = 0
            for document in documents:
                for sentence_index in range(len(document.annotations(self.sentence_annotation))):
                    metadata[i]["sentence_index"] = sentence_index
                    i += 1

        return input_, metadata, expanded_documents

    def encode_target(
        self, documents: List[Document], input_: List[Dict[str, Any]]
    ) -> Union[List[List[int]], List[Dict[str, Any]]]:
        target = []
        if self.single_sentence:
            i = 0
            for document in documents:
                entities = document.annotations(self.entity_annotation)
                sentences = document.annotations(self.sentence_annotation)

                for sentence in sentences:
                    word_ids = input_[i].word_ids()
                    label_ids = [
                        self.label_pad_token_id if word_ids[j] is None else self.label_to_id["O"]
                        for j in range(len(word_ids))
                    ]

                    for entity in entities:
                        if entity.start < sentence.start or entity.end > sentence.end:
                            continue

                        entity_start = entity.start - sentence.start
                        entity_end = entity.end - sentence.start

                        start_idx = input_[i].char_to_token(entity_start)
                        end_idx = input_[i].char_to_token(entity_end - 1)
                        for j in range(start_idx, end_idx + 1):
                            prefix = "B" if j == start_idx else "I"
                            label_ids[j] = self.label_to_id[f"{prefix}-{entity.label}"]

                    target.append(label_ids)
                    i += 1
        else:
            for i, document in enumerate(documents):
                word_ids = input_[i].word_ids()
                label_ids = [
                    self.label_pad_token_id if word_ids[j] is None else self.label_to_id["O"]
                    for j in range(len(word_ids))
                ]

                entities = document.annotations(self.entity_annotation)

                for entity in entities:
                    start_idx = input_[i].char_to_token(entity.start)
                    end_idx = input_[i].char_to_token(entity.end - 1)
                    for j in range(start_idx, end_idx + 1):
                        prefix = "B" if j == start_idx else "I"
                        label_ids[j] = self.label_to_id[f"{prefix}-{entity.label}"]

                target.append(label_ids)

        return target

    def decode_output(self, output: Dict[str, Any]) -> List[Dict[str, Any]]:
        logits = output["logits"]
        probabilities = F.softmax(logits, dim=-1).detach().cpu().numpy()
        indices = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        tags = [[self.id_to_label[e] for e in b] for b in indices]
        return [{"tags": t, "probabilities": p} for t, p in zip(tags, probabilities)]

    def decoded_output_to_annotations(
        self,
        decoded_output: Dict[str, Any],
        encoding: TaskEncoding,
    ) -> Iterator[Tuple[str, Annotation]]:
        if self.single_sentence:
            document = encoding.document
            metadata = encoding.metadata

            sentence = document.annotations(self.sentence_annotation)[
                metadata["sentence_index"]
            ]

            tag_sequence = [
                "O" if stm else tag
                for tag, stm in zip(decoded_output["tags"], metadata["special_tokens_mask"])
            ]

            spans = bio_tags_to_spans(tag_sequence)
            for label, (start, end) in spans:
                yield (
                    self.entity_annotation,
                    LabeledSpan(
                        sentence.start + metadata["offset_mapping"][start][0],
                        sentence.start + metadata["offset_mapping"][end][1],
                        label,
                    ),
                )
        else:
            metadata = encoding.metadata

            tag_sequence = [
                "O" if stm else tag
                for tag, stm in zip(decoded_output["tags"], metadata["special_tokens_mask"])
            ]

            spans = bio_tags_to_spans(tag_sequence)
            for label, (start, end) in spans:
                yield (
                    self.entity_annotation,
                    LabeledSpan(
                        metadata["offset_mapping"][start][0],
                        metadata["offset_mapping"][end][1],
                        label,
                    ),
                )

    def collate(self, encodings: List[TaskEncoding]) -> Dict[str, Any]:
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

        if encodings[0].target is None:
            return input_, None, metadata, documents

        target = [encoding.target for encoding in encodings]

        sequence_length = torch.tensor(input_["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            target = [
                list(t) + [self.label_pad_token_id] * (sequence_length - len(t)) for t in target
            ]
        else:
            target = [
                [self.label_pad_token_id] * (sequence_length - len(t)) + list(t) for t in target
            ]

        input_ = {k: torch.tensor(v, dtype=torch.int64) for k, v in input_.items()}
        target = torch.tensor(target, dtype=torch.int64)

        return input_, target, metadata, documents
