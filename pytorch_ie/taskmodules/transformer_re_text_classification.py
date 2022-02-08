from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, TypedDict, Union

import numpy as np
import torch
from transformers import AutoTokenizer
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy

from pytorch_ie.data.document import Annotation, BinaryRelation, Document, LabeledSpan
from pytorch_ie.data.span_utils import is_contained_in
from pytorch_ie.models import (
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

TransformerReTextClassificationInputEncoding = Dict[str, Any]
TransformerReTextClassificationTargetEncoding = List[int]

TransformerReTextClassificationTaskEncoding = TaskEncoding[
    TransformerReTextClassificationInputEncoding, TransformerReTextClassificationTargetEncoding
]
TransformerReTextClassificationTaskOutput = TypedDict(
    "TransformerReTextClassificationTaskOutput",
    {"labels": List[str], "probabilities": List[float]},
    total=False,
)

_TransformerReTextClassificationTaskModule = TaskModule[
    # _InputEncoding, _TargetEncoding, _TaskBatchEncoding, _ModelBatchOutput, _TaskOutput
    TransformerReTextClassificationInputEncoding,
    TransformerReTextClassificationTargetEncoding,
    TransformerTextClassificationModelStepBatchEncoding,
    TransformerTextClassificationModelBatchOutput,
    TransformerReTextClassificationTaskOutput,
]


def _create_argument_markers(
    entity_labels: List[str], add_type_to_marker: bool
) -> Dict[Union[Tuple[str, str, str], Tuple[str, str]], str]:
    argument_markers: Dict[Union[Tuple[str, str, str], Tuple[str, str]], str] = {}
    for arg_type in ["head", "tail"]:
        is_head = arg_type == "head"

        for arg_pos in ["start", "end"]:
            is_start = arg_pos == "start"

            if add_type_to_marker:
                for entity_type in entity_labels:
                    marker = f"[{'' if is_start else '/'}{'H' if is_head else 'T'}:{entity_type}]"
                    argument_markers[(arg_type, arg_pos, entity_type)] = marker
            else:
                marker = f"[{'' if is_start else '/'}{'H' if is_head else 'T'}]"
                argument_markers[(arg_type, arg_pos)] = marker

    return argument_markers


class TransformerRETextClassificationTaskModule(_TransformerReTextClassificationTaskModule):
    """
    Marker based relation extraction. This taskmodule prepares the input token ids in such a way
    that before and after the candidate head and tail entities special marker tokens are inserted.
    Then, the modified token ids can be simply passed into a transformer based text classifier model.

    parameters:

        partition_annotation: str, optional. If specified, LabeledSpan annotations with this name are
            expected to define partitions of the document that will be processed individually, e.g. sentences
            or sections of the document text.
        none_label: str, defaults to "no_relation". The relation label that indicate dummy/negative relations.
            Predicted relations with that label will not be added to the document(s).

        TODO: add remaining parameters
    """

    def __init__(
        self,
        tokenizer_name_or_path: str,
        entity_annotation: str = "entities",
        relation_annotation: str = "relations",
        partition_annotation: Optional[str] = None,
        none_label: str = "no_relation",
        padding: Union[bool, str, PaddingStrategy] = True,
        truncation: Union[bool, str, TruncationStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        multi_label: bool = False,
        label_to_id: Optional[Dict[str, int]] = None,
        add_type_to_marker: bool = False,
        single_argument_pair: bool = True,
        append_markers: bool = False,
        entity_labels: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            tokenizer_name_or_path=tokenizer_name_or_path,
            entity_annotation=entity_annotation,
            relation_annotation=relation_annotation,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            multi_label=multi_label,
            add_type_to_marker=add_type_to_marker,
            single_argument_pair=single_argument_pair,
            append_markers=append_markers,
            entity_labels=entity_labels,
            partition_annotation=partition_annotation,
            none_label=none_label,
        )

        self.entity_annotation = entity_annotation
        self.relation_annotation = relation_annotation
        self.padding = padding
        self.truncation = truncation
        self.label_to_id = label_to_id or {}
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.multi_label = multi_label
        self.add_type_to_marker = add_type_to_marker
        self.single_argument_pair = single_argument_pair
        self.append_markers = append_markers
        self.entity_labels = entity_labels
        self.partition_annotation = partition_annotation
        self.none_label = none_label

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

        self.argument_markers = None

        # this is the case when we load an already prepared taskmodule
        if self.entity_labels is not None:
            self.argument_markers = _create_argument_markers(
                entity_labels=self.entity_labels, add_type_to_marker=self.add_type_to_marker
            )
            # do not sort here to keep order from loaded taskmodule config
            self.tokenizer.add_tokens(list(self.argument_markers.values()), special_tokens=True)

    def _config(self) -> Dict[str, Any]:
        config = super()._config()
        config["label_to_id"] = self.label_to_id
        config["entity_labels"] = self.entity_labels
        return config

    def prepare(self, documents: List[Document]) -> None:
        entity_labels = set()
        relation_labels = set()
        for document in documents:
            entities = document.span_annotations(self.entity_annotation)
            relations = document.relation_annotations(self.relation_annotation)

            if self.add_type_to_marker:
                for entity in entities:
                    entity_labels.update(entity.labels)

            for relation in relations:
                relation_labels.update(relation.labels)

        if self.none_label in relation_labels:
            relation_labels.remove(self.none_label)

        self.label_to_id = {label: i + 1 for i, label in enumerate(sorted(relation_labels))}
        self.label_to_id[self.none_label] = 0

        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

        self.entity_labels = sorted(entity_labels)
        argument_markers = _create_argument_markers(
            entity_labels=self.entity_labels, add_type_to_marker=self.add_type_to_marker
        )
        # Sort argument markers by value to ensure that added tokens are in a reproducible order.
        # Note: To maintain backwards compatibility, the argument markers are not sorted when loading from a saved
        # taskmodule!
        self.argument_markers = dict(sorted(argument_markers.items(), key=lambda kv: kv[1]))
        self.tokenizer.add_tokens(list(self.argument_markers.values()), special_tokens=True)

    def encode_input(
        self, documents: List[Document]
    ) -> Tuple[
        List[TransformerReTextClassificationInputEncoding],
        List[Metadata],
        Optional[List[Document]],
    ]:
        assert (
            self.argument_markers is not None
        ), f"No argument markers available, was `prepare` already called?"
        argument_markers_to_id = {
            marker: self.tokenizer.vocab[marker] for marker in self.argument_markers.values()
        }
        input_encoding = []
        metadata = []
        new_documents = []

        for document in documents:
            entities = document.span_annotations(self.entity_annotation)
            relations = document.relation_annotations(self.relation_annotation)
            existing_head_tail = {(relation.head, relation.tail) for relation in relations}

            if self.partition_annotation is not None:
                partitions = document.span_annotations(self.partition_annotation)
            else:
                # use single dummy partition
                partitions = [LabeledSpan(start=0, end=len(document.text), label="FULL_DOCUMENT")]

            for partition_idx, partition in enumerate(partitions):
                encoding = self.tokenizer(
                    document.text[partition.start : partition.end],
                    padding=False,
                    truncation=self.truncation,
                    max_length=self.max_length,
                    is_split_into_words=False,
                    return_offsets_mapping=False,
                    # TODO: use this for windowing
                    # add_special_tokens=False,
                )

                head: LabeledSpan
                for head in entities:
                    if not is_contained_in(
                        (head.start, head.end), (partition.start, partition.end)
                    ):
                        continue

                    head_start = encoding.char_to_token(head.start - partition.start)
                    head_end = encoding.char_to_token(head.end - partition.start - 1)

                    if head_start is None or head_end is None:
                        continue

                    tail: LabeledSpan
                    for tail in entities:
                        if not is_contained_in(
                            (tail.start, tail.end), (partition.start, partition.end)
                        ):
                            continue

                        if head == tail:
                            continue

                        if relations and ((head, tail) not in existing_head_tail):
                            continue

                        tail_start = encoding.char_to_token(tail.start - partition.start)
                        tail_end = encoding.char_to_token(tail.end - partition.start - 1)

                        if tail_start is None or tail_end is None:
                            continue

                        # TODO: do windowing here!
                        if self.add_type_to_marker:
                            if head.is_multilabel:
                                raise NotImplementedError

                            head_start_marker = argument_markers_to_id[
                                self.argument_markers[("head", "start", head.label_single)]
                            ]
                            head_end_marker = argument_markers_to_id[
                                self.argument_markers[("head", "end", head.label_single)]
                            ]
                            if tail.is_multilabel:
                                raise NotImplementedError
                            tail_start_marker = argument_markers_to_id[
                                self.argument_markers[("tail", "start", tail.label_single)]
                            ]
                            tail_end_marker = argument_markers_to_id[
                                self.argument_markers[("tail", "end", tail.label_single)]
                            ]
                        else:
                            head_start_marker = argument_markers_to_id[
                                self.argument_markers[("head", "start")]
                            ]
                            head_end_marker = argument_markers_to_id[
                                self.argument_markers[("head", "end")]
                            ]
                            tail_start_marker = argument_markers_to_id[
                                self.argument_markers[("tail", "start")]
                            ]
                            tail_end_marker = argument_markers_to_id[
                                self.argument_markers[("tail", "end")]
                            ]

                        head_items = (head_start, head_end + 1, head_start_marker, head_end_marker)
                        tail_items = (tail_start, tail_end + 1, tail_start_marker, tail_end_marker)

                        head_first = head_start < tail_start
                        first, second = (
                            (head_items, tail_items) if head_first else (tail_items, head_items)
                        )

                        first_start, first_end, first_start_marker, first_end_marker = first
                        second_start, second_end, second_start_marker, second_end_marker = second

                        input_ids = encoding["input_ids"]

                        first_tokens = input_ids[first_start:first_end]
                        second_tokens = input_ids[second_start:second_end]

                        new_input_ids = (
                            input_ids[:first_start]
                            + [first_start_marker]
                            + first_tokens
                            + [first_end_marker]
                            + input_ids[first_end:second_start]
                            + [second_start_marker]
                            + second_tokens
                            + [second_end_marker]
                            + input_ids[second_end:]
                        )

                        new_head_start = new_input_ids.index(head_start_marker)
                        new_head_end = new_input_ids.index(head_end_marker)
                        new_tail_start = new_input_ids.index(tail_start_marker)
                        new_tail_end = new_input_ids.index(tail_end_marker)

                        # TODO: add special tokens here (windowing)
                        input_encoding.append({"input_ids": new_input_ids})
                        new_documents.append(document)
                        doc_metadata = {
                            "head": head,
                            "tail": tail,
                            "head_offset": (new_head_start, new_head_end),
                            "tail_offset": (new_tail_start, new_tail_end),
                        }
                        metadata.append(doc_metadata)

        return input_encoding, metadata, new_documents

    def encode_target(
        self,
        documents: List[Document],
        input_encodings: List[TransformerReTextClassificationInputEncoding],
        metadata: List[Metadata],
    ) -> List[TransformerReTextClassificationTargetEncoding]:

        target: List[TransformerReTextClassificationTargetEncoding] = []
        for i, document in enumerate(documents):
            meta = metadata[i]

            relations = document.relation_annotations(self.relation_annotation)

            head_tail_to_labels = {
                (relation.head, relation.tail): relation.labels for relation in relations
            }

            labels = head_tail_to_labels.get((meta["head"], meta["tail"]), [self.none_label])
            label_ids = [self.label_to_id[label] for label in labels]
            target.append(label_ids)

        return target

    def unbatch_output(
        self, output: TransformerTextClassificationModelBatchOutput
    ) -> Sequence[TransformerReTextClassificationTaskOutput]:
        logits = output["logits"]

        output_label_probs = logits.sigmoid() if self.multi_label else logits.softmax(dim=-1)
        output_label_probs = output_label_probs.detach().cpu().numpy()

        decoded_output = []
        if self.multi_label:
            raise NotImplementedError
        else:
            label_ids = np.argmax(output_label_probs, axis=-1)
            for batch_idx, label_id in enumerate(label_ids):
                label = self.id_to_label[label_id]
                prob = float(output_label_probs[batch_idx, label_id])
                result: TransformerReTextClassificationTaskOutput = {
                    "labels": [label],
                    "probabilities": [prob],
                }
                decoded_output.append(result)

        return decoded_output

    def create_annotations_from_output(
        self,
        encoding: TransformerReTextClassificationTaskEncoding,
        output: TransformerReTextClassificationTaskOutput,
    ) -> Iterator[Tuple[str, Annotation]]:
        labels = output["labels"]
        probabilities = output["probabilities"]
        if labels != [self.none_label]:
            yield (
                self.relation_annotation,
                BinaryRelation(
                    head=encoding.metadata["head"],
                    tail=encoding.metadata["tail"],
                    label=labels if self.multi_label else labels[0],
                    score=probabilities if self.multi_label else probabilities[0],
                ),
            )

    def collate(
        self, encodings: List[TransformerReTextClassificationTaskEncoding]
    ) -> TransformerTextClassificationModelStepBatchEncoding:

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

        target_list: List[TransformerReTextClassificationTargetEncoding] = [
            encoding.target for encoding in encodings
        ]
        target = torch.tensor(target_list, dtype=torch.int64)

        if not self.multi_label:
            target = target.flatten()

        return input_, target
