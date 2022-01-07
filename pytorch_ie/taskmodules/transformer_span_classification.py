from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy, BatchEncoding

from pytorch_ie.data.document import Annotation, Document, LabeledSpan
from pytorch_ie.models import TransformerSpanClassificationModelBatchOutput
from pytorch_ie.taskmodules.taskmodule import (
    TaskEncoding,
    TaskModule,
    Metadata,
)

"""
workflow:
    Document
        -> (InputEncoding, TargetEncoding) -> TaskEncoding -> TaskBatchEncoding
            -> ModelBatchEncoding -> ModelBatchOutput
        -> TaskOutput
    -> Document
"""
TransformerSpanClassificationInputEncoding = Dict[str, Any]
TransformerSpanClassificationTargetEncoding = List[int]
TransformerSpanClassificationTaskEncoding = TaskEncoding[
    TransformerSpanClassificationInputEncoding,
    TransformerSpanClassificationTargetEncoding
]
TransformerSpanClassificationTaskBatchEncoding = Tuple[
    Dict[str, Tensor],
    Optional[List[TransformerSpanClassificationTargetEncoding]],
    List[Metadata],
    List[Document]
]
TransformerSpanClassificationTaskOutput = Dict[str, Any]
_TransformerSpanClassificationTaskModule = TaskModule[
    # _InputEncoding, _TargetEncoding, _TaskBatchEncoding, _ModelBatchOutput, _TaskOutput
    TransformerSpanClassificationInputEncoding,
    TransformerSpanClassificationTargetEncoding,
    TransformerSpanClassificationTaskBatchEncoding,
    TransformerSpanClassificationModelBatchOutput,
    TransformerSpanClassificationTaskOutput,
]


class TransformerSpanClassificationTaskModule(_TransformerSpanClassificationTaskModule):
    def __init__(
        self,
        tokenizer_name_or_path: str,
        entity_annotation: str = "entities",
        single_sentence: bool = False,
        sentence_annotation: str = "sentences",
        padding: Union[bool, str, PaddingStrategy] = True,
        truncation: Union[bool, str, TruncationStrategy] = True,
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
            self.label_to_id[label] = current_id
            current_id += 1

        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

    def encode_input(
        self, documents: List[Document]
    ) -> Tuple[List[TransformerSpanClassificationInputEncoding], Optional[List[Metadata]], Optional[List[Document]]]:
        expanded_documents = None
        if self.single_sentence:
            sent: LabeledSpan
            input_ = [
                self.tokenizer(
                    doc.text[sent.start: sent.end],
                    padding=False,
                    truncation=self.truncation,
                    max_length=self.max_length,
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

        if self.single_sentence:
            i = 0
            for document in documents:
                for sentence_index in range(len(document.annotations(self.sentence_annotation))):
                    metadata[i]["sentence_index"] = sentence_index
                    i += 1

        return input_, metadata, expanded_documents

    def encode_target(
        self,
        documents: List[Document],
        input_encodings: List[TransformerSpanClassificationInputEncoding],
        metadata: Optional[List[Metadata]],
    ) -> List[TransformerSpanClassificationTargetEncoding]:
        input_encodings: List[BatchEncoding]
        target = []
        if self.single_sentence:
            for i, document in enumerate(documents):
                entities = document.annotations(self.entity_annotation)
                sentence_idx = metadata[i]["sentence_index"]
                sentence: LabeledSpan = document.annotations(self.sentence_annotation)[sentence_idx]

                label_ids = []
                entity: LabeledSpan
                for entity in entities:
                    if entity.start < sentence.start or entity.end > sentence.end:
                        continue

                    entity_start = entity.start - sentence.start
                    entity_end = entity.end - sentence.start

                    start_idx = input_encodings[i].char_to_token(entity_start)
                    end_idx = input_encodings[i].char_to_token(entity_end - 1)
                    label_ids.append((start_idx, end_idx, self.label_to_id[entity.label]))

                target.append(label_ids)
        else:
            for i, document in enumerate(documents):
                entities = document.annotations(self.entity_annotation)

                label_ids = []
                for entity in entities:
                    start_idx = input_encodings[i].char_to_token(entity.start)
                    end_idx = input_encodings[i].char_to_token(entity.end - 1)
                    label_ids.append((start_idx, end_idx, self.label_to_id[entity.label]))

                target.append(label_ids)

        return target

    def unbatch_output(self, output: TransformerSpanClassificationModelBatchOutput) \
        -> List[TransformerSpanClassificationTaskOutput]:
        logits = output["logits"]
        probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
        label_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()

        start_indices = output["start_indices"].detach().cpu().numpy()
        end_indices = output["end_indices"].detach().cpu().numpy()
        batch_indices = output["batch_indices"].detach().cpu().numpy()

        tags = [[] for _ in np.unique(batch_indices)]
        probabilities = [[] for _ in np.unique(batch_indices)]
        for start, end, batch_idx, label_id, prob in zip(
            start_indices, end_indices, batch_indices, label_ids, probs
        ):
            label = self.id_to_label[label_id]
            if label != "O":
                tags[batch_idx].append((label, (start, end)))
                probabilities[batch_idx].append(prob)

        # labels = [[self.id_to_label[e] for e in b] for b in label_ids]
        return [{"tags": t, "probabilities": p} for t, p in zip(tags, probabilities)]

    def create_annotations_from_output(
        self,
        output: Dict[str, Any],
        encoding: TransformerSpanClassificationTaskEncoding,
    ) -> Iterator[Tuple[str, Annotation]]:
        if self.single_sentence:
            document = encoding.document
            metadata = encoding.metadata

            sentence: LabeledSpan = document.annotations(self.sentence_annotation)[metadata["sentence_index"]]

            # tag_sequence = [
            #     "O" if stm else tag
            #     for tag, stm in zip(output["tags"], metadata["special_tokens_mask"])
            # ]

            # spans = bio_tags_to_spans(tag_sequence)
            spans = output["tags"]
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

            # tag_sequence = [
            #     "O" if stm else tag
            #     for tag, stm in zip(output["tags"], metadata["special_tokens_mask"])
            # ]

            # spans = bio_tags_to_spans(tag_sequence)
            spans = output["tags"]
            for label, (start, end) in spans:
                yield (
                    self.entity_annotation,
                    LabeledSpan(
                        metadata["offset_mapping"][start][0],
                        metadata["offset_mapping"][end][1],
                        label,
                    ),
                )

    def collate(
        self, encodings: List[TransformerSpanClassificationTaskEncoding]
    ) -> TransformerSpanClassificationTaskBatchEncoding:
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

        input_ = {k: torch.tensor(v, dtype=torch.int64) for k, v in input_.items()}

        return input_, target, metadata, documents
