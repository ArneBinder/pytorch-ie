import copy
import logging
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import BatchEncoding, TruncationStrategy

from pytorch_ie.annotations import LabeledSpan, Span
from pytorch_ie.core import TaskEncoding, TaskModule
from pytorch_ie.documents import TextDocument
from pytorch_ie.models.transformer_token_classification import (
    TransformerTokenClassificationModelBatchOutput,
    TransformerTokenClassificationModelStepBatchEncoding,
)
from pytorch_ie.utils.span import (
    bio_tags_to_spans,
    convert_span_annotations_to_tag_sequence,
    get_char_to_token_mapper,
    get_special_token_mask,
    has_overlap,
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
TransformerTokenClassificationInputEncoding = Union[Dict[str, Any], BatchEncoding]
TransformerTokenClassificationTargetEncoding = Sequence[int]

TransformerTokenClassificationTaskEncoding = TaskEncoding[
    TextDocument,
    TransformerTokenClassificationInputEncoding,
    TransformerTokenClassificationTargetEncoding,
]
TransformerTokenClassificationTaskOutput = Dict[str, Any]

_TransformerTokenClassificationTaskModule = TaskModule[
    # _InputEncoding, _TargetEncoding, _TaskBatchEncoding, _ModelBatchOutput, _TaskOutput
    TextDocument,
    TransformerTokenClassificationInputEncoding,
    TransformerTokenClassificationTargetEncoding,
    TransformerTokenClassificationModelStepBatchEncoding,
    TransformerTokenClassificationModelBatchOutput,
    TransformerTokenClassificationTaskOutput,
]

logger = logging.getLogger(__name__)


@TaskModule.register()
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
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
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

    def prepare(self, documents: Sequence[TextDocument]) -> None:
        labels = set()
        for document in documents:
            entities: Sequence[LabeledSpan] = document[self.entity_annotation]

            for entity in entities:
                labels.add(entity.label)
                # labels.update(entity.label)

        self.label_to_id["O"] = 0
        current_id = 1
        for label in sorted(labels):
            for prefix in ["B", "I"]:
                self.label_to_id[f"{prefix}-{label}"] = current_id
                current_id += 1

        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

    def encode_text(
        self, text, partition: Optional[Span] = None, add_special_tokens: bool = True
    ) -> BatchEncoding:
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
        document: TextDocument,
        is_training: bool = False,
    ) -> Optional[
        Union[
            TransformerTokenClassificationTaskEncoding,
            Sequence[TransformerTokenClassificationTaskEncoding],
        ]
    ]:
        partitions: Sequence[Optional[Span]]
        if self.partition_annotation is not None:
            partitions = document[self.partition_annotation]
        else:
            partitions = [None]

        task_encodings: List[TransformerTokenClassificationTaskEncoding] = []
        for partition_index, partition in enumerate(partitions):
            add_special_tokens = self.max_window is None
            inputs: BatchEncoding = self.encode_text(
                text=document.text, partition=partition, add_special_tokens=add_special_tokens
            )

            metadata = {
                "offset_mapping": inputs.pop("offset_mapping"),
                "special_tokens_mask": inputs.pop("special_tokens_mask"),
                "char_to_token_mapper": inputs.char_to_token,
            }

            if partition is not None:
                metadata["sentence_index"] = partition_index

            if self.max_window is None:
                task_encodings.append(
                    TaskEncoding(
                        document=document,
                        inputs=inputs,
                        metadata=metadata,
                    )
                )
            else:
                offset_mapping = metadata.pop("offset_mapping")
                # The actual number of tokens will be lower than max_window because we add the default special
                # tokens later on (e.g. CLS and SEP).
                max_window = self.max_window - self.tokenizer.num_special_tokens_to_add()
                token_ids = inputs["input_ids"]
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
                    window_inputs = {"input_ids": new_input_ids}
                    # for now, we copy just to keep "sentence_index"
                    window_metadata = copy.deepcopy(metadata)
                    window_metadata["special_tokens_mask"] = new_special_tokens_mask
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
                            current_offset_mapping.append(offset_mapping_without_special_tokens[j])
                            j += 1
                    window_metadata["offset_mapping"] = current_offset_mapping
                    char_to_token_mapping: Dict[int, int] = {}
                    for token_idx, (char_start, char_end) in enumerate(current_offset_mapping):
                        for char_idx in range(char_start, char_end):
                            char_to_token_mapping[char_idx] = token_idx
                    window_metadata["char_to_token_mapper"] = get_char_to_token_mapper(
                        char_to_token_mapping=char_to_token_mapping,
                        char_start=offset_mapping_without_special_tokens[0][0],
                        char_end=offset_mapping_without_special_tokens[-1][1],
                    )
                    # new_metadata["window_tokens"] = token_slice
                    window_metadata["window_labels"] = (
                        position_with_special_tokens[label_offset_slice[0]],
                        # we have to look up the actual index, not the pythonic end position
                        position_with_special_tokens[label_offset_slice[1] - 1] + 1,
                    )

                    task_encodings.append(
                        TaskEncoding(
                            document=document,
                            inputs=window_inputs,
                            metadata=window_metadata,
                        )
                    )

        return task_encodings

    def encode_target(
        self,
        task_encoding: TransformerTokenClassificationTaskEncoding,
    ) -> TransformerTokenClassificationTargetEncoding:
        metadata = task_encoding.metadata
        document = task_encoding.document

        entities: Sequence[LabeledSpan] = document[self.entity_annotation]

        partition = None
        if self.partition_annotation is not None:
            partition_index = metadata["sentence_index"]
            partitions = document[self.partition_annotation]
            partition = partitions[partition_index]
        tag_sequence = convert_span_annotations_to_tag_sequence(
            spans=entities,
            special_tokens_mask=metadata["special_tokens_mask"],
            char_to_token_mapper=metadata["char_to_token_mapper"],
            partition=partition,
            statistics=None,
        )

        # exclude labels that are out of the window (when overlap is used)
        window_labels = metadata.get("window_labels")
        if window_labels is not None:
            tag_sequence[0 : window_labels[0]] = [None] * window_labels[0]
            tag_sequence[window_labels[1] :] = [None] * len(tag_sequence[window_labels[1] :])

        targets = [
            self.label_to_id[tag] if tag is not None else self.label_pad_token_id
            for tag in tag_sequence
        ]

        return targets

    def unbatch_output(
        self, model_output: TransformerTokenClassificationModelBatchOutput
    ) -> Sequence[TransformerTokenClassificationTaskOutput]:
        logits = model_output["logits"]
        probabilities = F.softmax(logits, dim=-1).detach().cpu().numpy()
        indices = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        tags = [[self.id_to_label[e] for e in b] for b in indices]
        return [{"tags": t, "probabilities": p} for t, p in zip(tags, probabilities)]

    def create_annotations_from_output(
        self,
        task_encoding: TransformerTokenClassificationTaskEncoding,
        task_output: TransformerTokenClassificationTaskOutput,
    ) -> Iterator[Tuple[str, LabeledSpan]]:

        offset = 0
        if self.partition_annotation is not None:
            partitions = task_encoding.document[self.partition_annotation]
            offset = partitions[task_encoding.metadata["sentence_index"]].start

        tag_sequence = [
            "O" if is_special_token else tag
            for tag, is_special_token in zip(
                task_output["tags"], task_encoding.metadata["special_tokens_mask"]
            )
        ]

        spans = bio_tags_to_spans(
            tag_sequence, include_ill_formed=self.include_ill_formed_predictions
        )
        for label, (start, end) in spans:
            if "window_labels" in task_encoding.metadata:
                # Take only spans into account that are at least partly in the window. The model was not
                # trained to correctly predict spans that are just in the context.
                # NOTE: The "end" index is exclusive, but encoding.metadata["window_labels"][1] is inclusive!
                if not has_overlap((start, end + 1), task_encoding.metadata["window_labels"]):
                    continue
            yield (
                self.entity_annotation,
                LabeledSpan(
                    task_encoding.metadata["offset_mapping"][start][0] + offset,
                    task_encoding.metadata["offset_mapping"][end][1] + offset,
                    label,
                ),
            )

    def collate(
        self, task_encodings: Sequence[TransformerTokenClassificationTaskEncoding]
    ) -> TransformerTokenClassificationModelStepBatchEncoding:
        input_features = [task_encoding.inputs for task_encoding in task_encodings]

        inputs = self.tokenizer.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if not task_encodings[0].has_targets:
            return inputs, None

        target_list: List[TransformerTokenClassificationTargetEncoding] = [
            task_encoding.targets for task_encoding in task_encodings
        ]

        sequence_length = torch.tensor(inputs["input_ids"]).shape[1]
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

        inputs = {k: torch.tensor(v, dtype=torch.int64) for k, v in inputs.items()}
        targets = torch.tensor(target_list_padded, dtype=torch.int64)

        return inputs, targets
