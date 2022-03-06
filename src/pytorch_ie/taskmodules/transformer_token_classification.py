import copy
import json
import logging
from collections import Counter, defaultdict
from typing import Any, DefaultDict, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import BatchEncoding, TruncationStrategy

from pytorch_ie.data.document import Annotation, Document, LabeledSpan
from pytorch_ie.models.transformer_token_classification import (
    TransformerTokenClassificationModelBatchOutput,
    TransformerTokenClassificationModelStepBatchEncoding,
)
from pytorch_ie.taskmodules.taskmodule import Metadata, TaskEncoding, TaskModule
from pytorch_ie.utils.span import (
    bio_tags_to_spans,
    convert_span_annotations_to_tag_sequence,
    get_char_to_token_mapper,
    get_special_token_mask,
)
from pytorch_ie.utils.window import enumerate_windows

"""
workflow:
    Document
        -> (InputEncoding, TargetEncoding) -> TaskEncoding -> TaskBatchEncoding
            -> ModelBatchEncoding -> ModelBatchOutput
        -> TaskOutput
    -> Document
"""
TransformerTokenClassificationInputEncoding = BatchEncoding
TransformerTokenClassificationTargetEncoding = List[int]

TransformerTokenClassificationTaskEncoding = TaskEncoding[
    TransformerTokenClassificationInputEncoding, TransformerTokenClassificationTargetEncoding
]
TransformerTokenClassificationTaskOutput = Dict[str, Any]

_TransformerTokenClassificationTaskModule = TaskModule[
    # _InputEncoding, _TargetEncoding, _TaskBatchEncoding, _ModelBatchOutput, _TaskOutput
    TransformerTokenClassificationInputEncoding,
    TransformerTokenClassificationTargetEncoding,
    TransformerTokenClassificationModelStepBatchEncoding,
    TransformerTokenClassificationModelBatchOutput,
    TransformerTokenClassificationTaskOutput,
]

logger = logging.getLogger(__name__)


class TransformerTokenClassificationTaskModule(_TransformerTokenClassificationTaskModule):
    def __init__(
        self,
        tokenizer_name_or_path: str,
        entity_annotation: str = "entities",
        partition_annotation: Optional[str] = None,
        padding: Union[bool, str, PaddingStrategy] = True,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        label_pad_token_id: int = -100,
        label_to_id: Optional[Dict[str, int]] = None,
        max_window: Optional[int] = None,
        window_overlap: int = 0,
        show_statistics: bool = False,
        include_ill_formed_predictions: bool = True,
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
        self.max_window = max_window
        self.window_overlap = window_overlap
        self.show_statistics = show_statistics
        self.include_ill_formed_predictions = include_ill_formed_predictions

    def _config(self) -> Dict[str, Any]:
        config = super()._config()
        config["label_to_id"] = self.label_to_id
        return config

    def prepare(self, documents: List[Document]) -> None:
        labels = set()
        for document in documents:
            entities = document.annotations.spans[self.entity_annotation]

            for entity in entities:
                labels.update(entity.labels)

        self.label_to_id["O"] = 0
        current_id = 1
        for label in sorted(labels):
            for prefix in ["B", "I"]:
                self.label_to_id[f"{prefix}-{label}"] = current_id
                current_id += 1

        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

    def encode_text(
        self, text, partition: Optional[LabeledSpan] = None, add_special_tokens: bool = True
    ):
        if self.partition_annotation is not None and partition is None:
            raise ValueError(f"partitioning is enabled, but no partition is provided")
        text_partition = text[partition.start : partition.end] if partition is not None else text
        return self.tokenizer(
            text_partition,
            padding=False,
            truncation=False,
            max_length=None,
            is_split_into_words=False,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            add_special_tokens=add_special_tokens,
        )

    def encode_input(
        self,
        documents: List[Document],
        is_training: bool = False,
    ) -> Tuple[
        List[TransformerTokenClassificationInputEncoding],
        List[Metadata],
        Optional[List[Document]],
    ]:
        metadata = []
        expanded_documents = []
        input_ = []
        for doc in documents:
            partitions: Sequence[Optional[LabeledSpan]]
            if self.partition_annotation is not None:
                partitions = doc.annotations.spans[self.partition_annotation]
            else:
                partitions = [None]

            for partition_index, partition in enumerate(partitions):
                add_special_tokens = self.max_window is None
                encoding = self.encode_text(
                    text=doc.text, partition=partition, add_special_tokens=add_special_tokens
                )
                current_metadata = {
                    "offset_mapping": encoding.pop("offset_mapping"),
                    "special_tokens_mask": encoding.pop("special_tokens_mask"),
                    "char_to_token_mapper": encoding.char_to_token,
                }
                if partition is not None:
                    current_metadata["sentence_index"] = partition_index
                if self.max_window is None:
                    metadata.append(current_metadata)
                    input_.append(encoding)
                    expanded_documents.append(doc)
                else:
                    offset_mapping = current_metadata.pop("offset_mapping")
                    # The actual number of tokens will be lower than max_window because we add the default special
                    # tokens later on (e.g. CLS and SEP).
                    max_window = self.max_window - self.tokenizer.num_special_tokens_to_add()
                    token_ids = encoding["input_ids"]
                    for token_slice, label_offset_slice in enumerate_windows(
                        sequence=token_ids, max_size=max_window, overlap=self.window_overlap
                    ):
                        start_idx, end_idx = token_slice
                        new_input_ids = self.tokenizer.build_inputs_with_special_tokens(
                            token_ids_0=token_ids[start_idx:end_idx]
                        )
                        new_special_tokens_mask = get_special_token_mask(
                            token_ids_0=new_input_ids, tokenizer=self.tokenizer
                        )
                        new_encoding = {"input_ids": new_input_ids}
                        # for now, we copy just to keep "sentence_index"
                        new_metadata = copy.deepcopy(current_metadata)
                        new_metadata["special_tokens_mask"] = new_special_tokens_mask
                        offset_mapping_without_special_tokens = offset_mapping[start_idx:end_idx]
                        j = 0
                        current_offset_mapping: List[Tuple[int, int]] = []
                        # this maps from positions without special tokens to positions with special tokens
                        position_with_special_tokens = {}
                        for i, is_special_token in enumerate(new_special_tokens_mask):
                            if is_special_token:
                                current_offset_mapping.append((0, 0))
                            else:
                                position_with_special_tokens[j] = i
                                current_offset_mapping.append(
                                    offset_mapping_without_special_tokens[j]
                                )
                                j += 1
                        new_metadata["offset_mapping"] = current_offset_mapping
                        char_to_token_mapping: Dict[int, int] = {}
                        for token_idx, (char_start, char_end) in enumerate(current_offset_mapping):
                            for char_idx in range(char_start, char_end):
                                char_to_token_mapping[char_idx] = token_idx
                        new_metadata["char_to_token_mapper"] = get_char_to_token_mapper(
                            char_to_token_mapping=char_to_token_mapping,
                            char_start=offset_mapping_without_special_tokens[0][0],
                            char_end=offset_mapping_without_special_tokens[-1][1],
                        )
                        # new_metadata["window_tokens"] = token_slice
                        new_metadata["window_labels"] = (
                            position_with_special_tokens[label_offset_slice[0]],
                            # we have to look up the actual index, not the pythonic end position
                            position_with_special_tokens[label_offset_slice[1] - 1] + 1,
                        )

                        metadata.append(new_metadata)
                        input_.append(new_encoding)
                        expanded_documents.append(doc)

        return input_, metadata, expanded_documents

    def encode_target(
        self,
        documents: List[Document],
        input_encodings: List[TransformerTokenClassificationInputEncoding],
        metadata: List[Metadata],
    ) -> List[TransformerTokenClassificationTargetEncoding]:
        target = []
        statistics: Optional[DefaultDict[str, Counter]] = (
            defaultdict(Counter) if self.show_statistics else None
        )
        for i, document in enumerate(documents):
            current_metadata = metadata[i]
            entities = document.annotations.spans[self.entity_annotation]
            partition = None
            if self.partition_annotation is not None:
                partition_index = current_metadata["sentence_index"]
                partitions = document.annotations.spans[self.partition_annotation]
                partition = partitions[partition_index]
            tag_sequence = convert_span_annotations_to_tag_sequence(
                spans=entities,
                special_tokens_mask=current_metadata["special_tokens_mask"],
                char_to_token_mapper=current_metadata["char_to_token_mapper"],
                partition=partition,
                statistics=statistics,
            )
            # exclude labels that are out of the window (when overlap is used)
            window_labels = current_metadata.get("window_labels")
            if window_labels is not None:
                tag_sequence[0 : window_labels[0]] = [None] * window_labels[0]
                tag_sequence[window_labels[1] :] = [None] * len(tag_sequence[window_labels[1] :])
            label_ids = [
                self.label_to_id[tag] if tag is not None else self.label_pad_token_id
                for tag in tag_sequence
            ]
            target.append(label_ids)

        if statistics is not None:
            logger.info(f"statistics:\n{json.dumps(statistics, indent=2)}")
        return target

    def unbatch_output(
        self, output: TransformerTokenClassificationModelBatchOutput
    ) -> Sequence[TransformerTokenClassificationTaskOutput]:
        logits = output["logits"]
        probabilities = F.softmax(logits, dim=-1).detach().cpu().numpy()
        indices = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        tags = [[self.id_to_label[e] for e in b] for b in indices]
        return [{"tags": t, "probabilities": p} for t, p in zip(tags, probabilities)]

    def create_annotations_from_output(
        self,
        encoding: TransformerTokenClassificationTaskEncoding,
        output: TransformerTokenClassificationTaskOutput,
    ) -> Iterator[Tuple[str, Annotation]]:

        offset = 0
        if self.partition_annotation is not None:
            partitions = encoding.document.annotations.spans[self.partition_annotation]
            offset = partitions[encoding.metadata["sentence_index"]].start

        tag_sequence = [
            "O" if is_special_token else tag
            for tag, is_special_token in zip(
                output["tags"], encoding.metadata["special_tokens_mask"]
            )
        ]

        spans = bio_tags_to_spans(
            tag_sequence, include_ill_formed=self.include_ill_formed_predictions
        )
        for label, (start, end) in spans:
            yield (
                self.entity_annotation,
                LabeledSpan(
                    encoding.metadata["offset_mapping"][start][0] + offset,
                    encoding.metadata["offset_mapping"][end][1] + offset,
                    label,
                ),
            )

    def collate(
        self, encodings: List[TransformerTokenClassificationTaskEncoding]
    ) -> TransformerTokenClassificationModelStepBatchEncoding:
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

        target_list: List[TransformerTokenClassificationTargetEncoding] = [
            encoding.target for encoding in encodings
        ]

        sequence_length = torch.tensor(input_["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            target_list_padded = [
                list(t) + [self.label_pad_token_id] * (sequence_length - len(t))
                for t in target_list
            ]
        else:
            target_list_padded = [
                [self.label_pad_token_id] * (sequence_length - len(t)) + list(t)
                for t in target_list
            ]

        input_ = {k: torch.tensor(v, dtype=torch.int64) for k, v in input_.items()}
        target = torch.tensor(target_list_padded, dtype=torch.int64)

        return input_, target
