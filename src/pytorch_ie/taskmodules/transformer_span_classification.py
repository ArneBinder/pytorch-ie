import logging
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import BatchEncoding, TruncationStrategy

from pytorch_ie.data.document import Annotation, Document, LabeledSpan
from pytorch_ie.models.transformer_span_classification import (
    TransformerSpanClassificationModelBatchOutput,
    TransformerSpanClassificationModelStepBatchEncoding,
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

TransformerSpanClassificationInputEncoding = BatchEncoding
TransformerSpanClassificationTargetEncoding = List[Tuple[int, int, int]]

TransformerSpanClassificationTaskEncoding = TaskEncoding[
    TransformerSpanClassificationInputEncoding, TransformerSpanClassificationTargetEncoding
]
TransformerSpanClassificationTaskOutput = Dict[str, Any]

_TransformerSpanClassificationTaskModule = TaskModule[
    # _InputEncoding, _TargetEncoding, _TaskBatchEncoding, _ModelBatchOutput, _TaskOutput
    TransformerSpanClassificationInputEncoding,
    TransformerSpanClassificationTargetEncoding,
    TransformerSpanClassificationModelStepBatchEncoding,
    TransformerSpanClassificationModelBatchOutput,
    TransformerSpanClassificationTaskOutput,
]


logger = logging.getLogger(__name__)


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
        super().__init__()
        self.save_hyperparameters()

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

    def _config(self) -> Dict[str, Any]:
        config = super()._config()
        config["label_to_id"] = self.label_to_id
        return config

    def prepare(self, documents: List[Document]) -> None:
        labels = set()
        for document in documents:
            entities = document.annotations.spans[self.entity_annotation]

            for entity in entities:
                # TODO: labels is a set, use update
                for label in entity.labels:
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
        List[TransformerSpanClassificationInputEncoding],
        List[Metadata],
        Optional[List[Document]],
    ]:
        input_ = []
        expanded_documents = []
        for doc in documents:
            if self.single_sentence:
                partitions = doc.annotations.spans[self.sentence_annotation]
            else:
                partitions = [LabeledSpan(start=0, end=len(doc.text), label="FULL_DOCUMENT")]
            for partition in partitions:
                encoding = self.tokenizer(
                    doc.text[partition.start : partition.end],
                    padding=False,
                    truncation=self.truncation,
                    max_length=self.max_length,
                    is_split_into_words=False,
                    return_offsets_mapping=True,
                    return_special_tokens_mask=True,
                )
                input_.append(encoding)
                expanded_documents.append(doc)

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
                for sentence_index in range(
                    len(document.annotations.spans[self.sentence_annotation])
                ):
                    metadata[i]["sentence_index"] = sentence_index
                    i += 1

        return input_, metadata, expanded_documents

    def encode_target(
        self,
        documents: List[Document],
        input_encodings: List[TransformerSpanClassificationInputEncoding],
        metadata: List[Metadata],
    ) -> List[TransformerSpanClassificationTargetEncoding]:
        target = []
        if self.single_sentence:
            for i, document in enumerate(documents):
                entities = document.annotations.spans[self.entity_annotation]
                sentence_idx = metadata[i]["sentence_index"]
                partitions = document.annotations.spans[self.sentence_annotation]
                assert (
                    partitions is not None
                ), f"document has no span annotations with name '{self.sentence_annotation}'"
                sentence = partitions[sentence_idx]

                label_ids = []
                entity: LabeledSpan
                for entity in entities:
                    if entity.start < sentence.start or entity.end > sentence.end:
                        continue

                    entity_start = entity.start - sentence.start
                    entity_end = entity.end - sentence.start

                    start_idx = input_encodings[i].char_to_token(entity_start)
                    end_idx = input_encodings[i].char_to_token(entity_end - 1)
                    # TODO: remove this is if case
                    if start_idx is None or end_idx is None:
                        logger.warning(
                            f"Entity annotation does not start or end with a token, it will be skipped: {entity}"
                        )
                        continue

                    label_ids.append((start_idx, end_idx, self.label_to_id[entity.label_single]))

                target.append(label_ids)
        else:
            for i, document in enumerate(documents):
                entities = document.annotations.spans[self.entity_annotation]
                label_ids = []
                for entity in entities:
                    start_idx = input_encodings[i].char_to_token(entity.start)
                    end_idx = input_encodings[i].char_to_token(entity.end - 1)
                    label_ids.append((start_idx, end_idx, self.label_to_id[entity.label_single]))

                target.append(label_ids)

        return target

    def unbatch_output(
        self, output: TransformerSpanClassificationModelBatchOutput
    ) -> Sequence[TransformerSpanClassificationTaskOutput]:
        logits = output["logits"]
        probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
        label_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()

        start_indices = output["start_indices"].detach().cpu().numpy()
        end_indices = output["end_indices"].detach().cpu().numpy()
        batch_indices = output["batch_indices"].detach().cpu().numpy()

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
        encoding: TransformerSpanClassificationTaskEncoding,
        output: TransformerSpanClassificationTaskOutput,
    ) -> Iterator[Tuple[str, Annotation]]:
        if self.single_sentence:
            document = encoding.document
            metadata = encoding.metadata
            partitions = document.annotations.spans[self.sentence_annotation]
            sentence = partitions[metadata["sentence_index"]]

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
    ) -> TransformerSpanClassificationModelStepBatchEncoding:
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

        target_list: List[TransformerSpanClassificationTargetEncoding] = [
            encoding.target for encoding in encodings
        ]
        input_ = {k: torch.tensor(v, dtype=torch.int64) for k, v in input_.items()}

        return input_, target_list
