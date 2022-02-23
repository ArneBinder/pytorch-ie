import copy
import functools
import json
import logging
from collections import Counter, defaultdict
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    Iterator,
    List,
    MutableSequence,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import BatchEncoding, TruncationStrategy

from pytorch_ie.data.document import Annotation, Document, LabeledSpan
from pytorch_ie.data.span_utils import bio_tags_to_spans
from pytorch_ie.models.transformer_token_classification import (
    TransformerTokenClassificationModelBatchOutput,
    TransformerTokenClassificationModelStepBatchEncoding,
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


def convert_span_annotations_to_tag_sequence(
    spans: List[LabeledSpan],
    special_tokens_mask: List[int],
    char_to_token_mapper: Callable[[int], Optional[int]],
    partition: Optional[LabeledSpan] = None,
    statistics: Optional[DefaultDict[str, Counter]] = None,
) -> MutableSequence[Optional[str]]:
    """
    Given a list of span annotations, a character position to token mapper (as obtained from
    batch_encoding.char_to_token) and a special tokens mask, create a sequence of tags with the length of the
    special tokens mask. For special token positions, None is returned as tag.
    If a partition is provided, only the tokens within that span are considered.
    For now, the BIO-encoding is used.
    Note: The spans are not allowed to overlap (will raise an exception).
    """
    tag_sequence = [
        None if special_tokens_mask[j] else "O" for j in range(len(special_tokens_mask))
    ]
    offset = partition.start if partition is not None else 0
    for span in spans:
        if partition is not None and (span.start < partition.start or span.end > partition.end):
            continue

        start_idx = char_to_token_mapper(span.start - offset)
        end_idx = char_to_token_mapper(span.end - 1 - offset)
        if start_idx is None or end_idx is None:
            if statistics is not None:
                statistics["skipped_unaligned"][span.label_single] += 1
            else:
                logger.warning(
                    f"Entity annotation does not start or end with a token, it will be skipped: {span}"
                )
            continue

        for j in range(start_idx, end_idx + 1):
            if tag_sequence[j] is not None and tag_sequence[j] != "O":
                # TODO: is ValueError a good exception type for this?
                raise ValueError(f"tag already assigned (current span has an overlap: {span})")
            prefix = "B" if j == start_idx else "I"
            tag_sequence[j] = f"{prefix}-{span.label_single}"

        if statistics is not None:
            statistics["added"][span.label_single] += 1

    return tag_sequence


def enumerate_windows(
    sequence: Sequence, max_size, overlap: int = 0
) -> Iterator[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Enumerate all windows as slices over a sequence, optionally with an overlap. Overlap is interpreted as number of
    tokens taken into account that are already part in another window. We also return label_offset_slice that defines
    the slice (with respect to the token_slice!) of tokens that are not available in another slice.
    """
    window_without_overlap = max_size - 2 * overlap
    for label_start_idx in range(overlap, len(sequence), window_without_overlap):
        token_start_idx = label_start_idx - overlap
        label_end_idx = min(label_start_idx + window_without_overlap, len(sequence))
        token_end_idx = min(label_end_idx + overlap, len(sequence))
        label_start_offset = label_start_idx - token_start_idx
        label_end_offset = label_end_idx - token_start_idx
        token_slice = (token_start_idx, token_end_idx)
        # also allow using previous/remaining entries as labels if we are at the beginning/end
        # to cover all entries exactly once in a label slice
        if token_start_idx == 0:
            label_start_offset = 0
        if token_end_idx == len(sequence):
            label_end_offset = token_end_idx - token_start_idx
        label_offset_slice = (label_start_offset, label_end_offset)
        yield token_slice, label_offset_slice


def get_special_token_mask(token_ids_0: List[int], tokenizer: PreTrainedTokenizer) -> List[int]:
    # TODO: check why we can not just use tokenizer.get_special_tokens_mask()
    #  (this checks if token_ids_1 is not None and raises an exception)

    # exclude unknown token id since this indicate a real input token
    special_ids = set(tokenizer.all_special_ids) - set([tokenizer.unk_token_id])
    return [1 if token_id in special_ids else 0 for token_id in token_ids_0]


def _char_to_token_mapper(c: int, char_to_token_mapping: Dict[int, int]) -> Optional[int]:
    return char_to_token_mapping.get(c, None)


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
            entities = document.span_annotations(self.entity_annotation)
            assert (
                entities is not None
            ), f"document has no span annotations with name '{self.entity_annotation}'"

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
        self, documents: List[Document]
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
                partitions_or_none = doc.span_annotations(self.partition_annotation)
                assert (
                    partitions_or_none
                ), f"document has no span annotations with name '{self.partition_annotation}'"
                partitions = partitions_or_none
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
                        new_metadata["char_to_token_mapper"] = functools.partial(
                            _char_to_token_mapper,
                            char_to_token_mapping=char_to_token_mapping,
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
            entities = document.span_annotations(self.entity_annotation)
            assert (
                entities
            ), f"document has no span annotations with name '{self.entity_annotation}'"
            partition = None
            if self.partition_annotation is not None:
                partition_index = current_metadata["sentence_index"]
                partitions = document.span_annotations(self.partition_annotation)
                assert (
                    partitions
                ), f"document has no span annotations with name '{self.partition_annotation}'"
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
            partitions = encoding.document.span_annotations(self.partition_annotation)
            assert (
                partitions
            ), f"document has no span annotations with name '{self.partition_annotation}'"
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
