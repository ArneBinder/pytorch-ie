import dataclasses
import json
import logging
from collections import Counter, defaultdict
from functools import cmp_to_key
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Type, Union

import torch
from pie_core import Annotation, AnnotationLayer, Document, TaskEncoding, TaskModule
from pie_core.taskmodule import InputEncoding, ModelBatchOutput, TargetEncoding, TaskBatchEncoding
from pie_core.utils.hydra import resolve_type
from pie_documents.annotations import BinaryRelation, LabeledSpan
from pie_documents.document.processing import token_based_document_to_text_based

# import for backwards compatibility (don't remove!)
from pie_documents.documents import (
    TextBasedDocument,
    TokenBasedDocument,
    TokenDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
)
from torch import Tensor
from torchmetrics import Metric
from transformers import AutoTokenizer, LogitsProcessorList, PreTrainedTokenizer
from typing_extensions import TypeAlias

from ..utils.document import tokenize_document
from .common import BatchableMixin, get_first_occurrence_index
from .metrics import (
    PrecisionRecallAndF1ForLabeledAnnotations,
    WrappedLayerMetricsWithUnbatchAndDecodeWithErrorsFunction,
)
from .pointer_network.annotation_encoder_decoder import (
    KEY_INVALID_CORRECT,
    BinaryRelationEncoderDecoder,
    LabeledSpanEncoderDecoder,
    SpanEncoderDecoderWithOffset,
)
from .pointer_network.logits_processor import (
    FinitizeLogitsProcessor,
    PrefixConstrainedLogitsProcessorWithMaximum,
)

logger = logging.getLogger(__name__)


DocumentType: TypeAlias = TextBasedDocument


@dataclasses.dataclass
class InputEncodingType(BatchableMixin):
    input_ids: List[int]
    attention_mask: List[int]


@dataclasses.dataclass
class LabelsAndOptionalConstraints(BatchableMixin):
    labels: List[int]
    constraints: Optional[List[List[int]]] = None

    @property
    def decoder_attention_mask(self) -> List[int]:
        return [1] * len(self.labels)


TargetEncodingType: TypeAlias = LabelsAndOptionalConstraints
TaskEncodingType: TypeAlias = TaskEncoding[
    DocumentType,
    InputEncodingType,
    TargetEncodingType,
]
TaskOutputType: TypeAlias = LabelsAndOptionalConstraints


def cmp_src_rel(v1: BinaryRelation, v2: BinaryRelation) -> int:
    if not all(isinstance(ann, LabeledSpan) for ann in [v1.head, v1.tail, v2.head, v2.tail]):
        raise Exception(f"expected LabeledSpan, but got: {v1}, {v2}")
    if v1.head.start == v2.head.start:  # v1[0]["from"] == v2[0]["from"]:
        return v1.tail.start - v2.tail.start  # v1[1]["from"] - v2[1]["from"]
    return v1.head.start - v2.head.start  # v1[0]["from"] - v2[0]["from"]


@TaskModule.register()
class PointerNetworkTaskModuleForEnd2EndRE(
    TaskModule[
        DocumentType,
        InputEncoding,
        TargetEncoding,
        TaskBatchEncoding,
        ModelBatchOutput,
        TaskOutputType,
    ],
):
    PREPARED_ATTRIBUTES = ["labels_per_layer"]
    REVERSED_RELATION_LABEL_SUFFIX = "_reversed"

    def __init__(
        self,
        tokenizer_name_or_path: str,
        # specific for this use case
        document_type: str = "pie_documents.documents.TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions",
        tokenized_document_type: str = "pie_documents.documents.TokenDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions",
        relation_layer_name: str = "binary_relations",
        add_reversed_relations: bool = False,
        symmetric_relations: Optional[List[str]] = None,
        none_label: str = "none",
        loop_dummy_relation_name: str = "loop",
        constrained_generation: bool = False,
        # generic pointer network
        label_tokens: Optional[Dict[str, str]] = None,
        label_representations: Optional[Dict[str, str]] = None,
        labels_per_layer: Optional[Dict[str, List[str]]] = None,
        exclude_labels_per_layer: Optional[Dict[str, List[str]]] = None,
        # target encoding
        create_constraints: bool = False,
        # tokenization
        tokenizer_init_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        partition_layer_name: Optional[str] = None,
        annotation_field_mapping: Optional[Dict[str, str]] = None,
        # logging
        log_first_n_examples: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        # tokenization
        self._document_type: Type[TextBasedDocument] = resolve_type(
            document_type, expected_super_type=TextBasedDocument
        )
        self._tokenized_document_type: Type[TokenBasedDocument] = resolve_type(
            tokenized_document_type, expected_super_type=TokenBasedDocument
        )
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            **(tokenizer_init_kwargs or {}),
        )
        self.annotation_field_mapping = annotation_field_mapping or dict()
        annotation_field_mapping_inv = {v: k for k, v in self.annotation_field_mapping.items()}
        if len(self.annotation_field_mapping) != len(annotation_field_mapping_inv):
            raise ValueError(
                f"inverted annotation_field_mapping is not unique. annotation_field_mapping: "
                f"{self.annotation_field_mapping}"
            )
        self.partition_layer_name = partition_layer_name

        # for this specific use case: end-to-end relation extraction
        self.relation_layer_name = relation_layer_name
        relation_layer_mapped = self.annotation_field_mapping.get(
            relation_layer_name, relation_layer_name
        )
        relation_layer_target = self.document_type.target_name(relation_layer_mapped)
        self.span_layer_name = annotation_field_mapping_inv.get(
            relation_layer_target, relation_layer_target
        )
        self.add_reversed_relations = add_reversed_relations
        self.symmetric_relations = set(symmetric_relations or [])
        self.none_label = none_label
        self.loop_dummy_relation_name = loop_dummy_relation_name
        self.constrained_generation = constrained_generation
        # will be set in _post_prepare()
        self.relation_encoder_decoder: BinaryRelationEncoderDecoder

        # collected in prepare(), if not passed in
        self.labels_per_layer = labels_per_layer
        self.exclude_labels_per_layer = exclude_labels_per_layer or {}

        # how to encode and decode the annotations
        self.bos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eos_token
        self.label_tokens = label_tokens or dict()
        self.label_representations = label_representations or dict()

        # target encoding
        self.create_constraints = create_constraints
        self.pad_values = {
            "input_ids": self.tokenizer.pad_token_id,
            "attention_mask": 0,
            "labels": self.target_pad_id,
            "decoder_attention_mask": 0,
            "constraints": -1,
        }
        self.dtypes = {
            "input_ids": torch.int64,
            "attention_mask": torch.int64,
            "labels": torch.int64,
            "decoder_attention_mask": torch.int64,
            "constraints": torch.int64,
        }

        # logging
        self.log_first_n_examples = log_first_n_examples

    @property
    def document_type(self) -> Type[TextBasedDocument]:
        return self._document_type

    @property
    def tokenized_document_type(self) -> Type[TokenBasedDocument]:
        return self._tokenized_document_type

    @property
    def layer_names(self) -> List[str]:
        return [self.span_layer_name, self.relation_layer_name]

    @property
    def special_targets(self) -> list[str]:
        return [self.bos_token, self.eos_token]

    @property
    def special_target2id(self) -> Dict[str, int]:
        return {target: idx for idx, target in enumerate(self.special_targets)}

    @property
    def target_pad_id(self) -> int:
        return self.special_target2id[self.eos_token]

    def configure_model_generation(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"no_repeat_ngram_size": 7}
        if self.constrained_generation:
            logits_processor = LogitsProcessorList()
            # PrefixConstrainedLogitsProcessorWithMaximum requires finite logits
            logits_processor.append(FinitizeLogitsProcessor())
            logits_processor.append(
                PrefixConstrainedLogitsProcessorWithMaximum(
                    prefix_allowed_tokens_fn=self._prefix_allowed_tokens_fn_with_maximum,
                    # use dummy value of 1, this is fine because num_beams affects only the value of batch_id
                    # which is not used in _prefix_allowed_tokens_fn_with_maximum()
                    num_beams=1,
                )
            )
            result["logits_processor"] = logits_processor
        return result

    def _prefix_allowed_tokens_fn_with_maximum(
        self, batch_id: int, input_ids: torch.LongTensor, maximum: int
    ) -> List[int]:
        # remove the first token (bos_token) and use unbatch_output to un-pad the label_ids
        label_ids_without_bos = input_ids[1:]
        if len(label_ids_without_bos) > 0:
            unpadded_label_ids = self.unbatch_output(
                {"labels": label_ids_without_bos.unsqueeze(0)}
            )[0].labels
        else:
            unpadded_label_ids = []
        _, _, remaining = self.relation_encoder_decoder.parse(encoding=unpadded_label_ids)
        # this is a binary mask
        constraint = self._build_constraint(
            previous_ids=remaining, input_len=maximum - self.pointer_offset
        )
        # convert to indices
        allowed_indices = torch.nonzero(constraint).squeeze(1)
        # convert to a list
        return allowed_indices.tolist()

    def add_reversed_relation_labels(self, relation_labels: Iterable[str]) -> Set[str]:
        result = set(relation_labels)
        for rel_label in set(relation_labels):
            if rel_label not in self.symmetric_relations:
                reversed_label = rel_label + self.REVERSED_RELATION_LABEL_SUFFIX
                if reversed_label in result:
                    raise ValueError(
                        f"reversed relation label {reversed_label} already exists in relation layer labels"
                    )
                result.add(reversed_label)
        return result

    def _prepare(self, documents: Sequence[DocumentType]) -> None:
        # collect all labels
        labels: Dict[str, Set[str]] = {layer_name: set() for layer_name in self.layer_names}
        for doc in documents:
            for layer_name in self.layer_names:
                exclude_labels = self.exclude_labels_per_layer.get(layer_name, [])
                labels[layer_name].update(
                    ac.label for ac in doc[layer_name] if ac.label not in exclude_labels
                )

        if self.add_reversed_relations:
            labels[self.relation_layer_name] = self.add_reversed_relation_labels(
                relation_labels=labels[self.relation_layer_name]
            )

        self.labels_per_layer = {
            # sort labels to ensure deterministic order
            layer_name: sorted(labels)
            for layer_name, labels in labels.items()
        }

    def construct_label_token(self, label: str) -> str:
        return self.label_tokens.get(label, f"<<{label}>>")

    def get_label_representation(self, label: str) -> str:
        return self.label_representations.get(label, label)

    def _post_prepare(self) -> None:
        # set up labels
        if self.labels_per_layer is None:
            raise Exception("labels_per_layer is not defined. Call prepare() first or pass it in.")
        self.labels: List[str] = [self.none_label]
        for layer_name in self.layer_names:
            self.labels.extend(self.labels_per_layer[layer_name])
        if len(set(self.labels)) != len(self.labels):
            raise Exception(f"labels are not unique: {self.labels}")

        # set up targets and ids
        self.targets: List[str] = self.special_targets + self.labels
        self.target2id: Dict[str, int] = {target: idx for idx, target in enumerate(self.targets)}

        # generic ids
        self.eos_id: int = self.target2id[self.eos_token]
        self.bos_id: int = self.target2id[self.bos_token]

        # span and relation ids
        self.span_ids: List[int] = [
            self.target2id[label] for label in self.labels_per_layer[self.span_layer_name]
        ]
        self.relation_ids: List[int] = [
            self.target2id[label] for label in self.labels_per_layer[self.relation_layer_name]
        ]
        # the none id is used for the dummy relation which models out-of-relation spans
        self.none_id: int = self.target2id[self.none_label]

        # helpers (same as targets / target2id, but only for labels)
        self.label2id: Dict[str, int] = {label: self.target2id[label] for label in self.labels}
        self.id2label: Dict[int, str] = {idx: label for label, idx in self.label2id.items()}
        self.label_ids: List[int] = [self.label2id[label] for label in self.labels]

        # annotation-encoder-decoders
        span_encoder_decoder = SpanEncoderDecoderWithOffset(
            offset=self.pointer_offset, exclusive_end=False
        )
        span_labels = self.labels_per_layer[self.span_layer_name]
        labeled_span_encoder_decoder = LabeledSpanEncoderDecoder(
            span_encoder_decoder=span_encoder_decoder,
            # restrict label2id to get better error messages
            label2id={label: idx for label, idx in self.label2id.items() if label in span_labels},
            mode="indices_label",
        )
        relation_labels = self.labels_per_layer[self.relation_layer_name] + [
            self.loop_dummy_relation_name,
            self.none_label,
        ]
        self.relation_encoder_decoder = BinaryRelationEncoderDecoder(
            head_encoder_decoder=labeled_span_encoder_decoder,
            tail_encoder_decoder=labeled_span_encoder_decoder,
            # restrict label2id to get better error messages
            label2id={
                label: idx for label, idx in self.label2id.items() if label in relation_labels
            },
            loop_dummy_relation_name=self.loop_dummy_relation_name,
            none_label=self.none_label,
            mode="tail_head_label",
        )

        label2token = {label: self.construct_label_token(label=label) for label in self.labels}
        if len(set(label2token.values())) != len(label2token):
            raise Exception(f"label2token values are not unique: {label2token}")

        already_in_vocab = [
            tok
            for tok in label2token.values()
            if self.tokenizer.convert_tokens_to_ids(tok) != self.tokenizer.unk_token_id
        ]
        if len(already_in_vocab) > 0:
            raise Exception(
                f"some special tokens to add (mapped label ids) are already in the tokenizer vocabulary, "
                f"this is not allowed: {already_in_vocab}. You may want to adjust the label2special_token mapping"
            )
        # sort by length, so that longer tokens are added first
        label_tokens_sorted = sorted(label2token.values(), key=lambda x: len(x), reverse=True)
        self.tokenizer.add_special_tokens(
            special_tokens_dict={"additional_special_tokens": label_tokens_sorted}
        )

        # target tokens are the special tokens plus the mapped label tokens
        self.target_tokens: List[str] = self.special_targets + [
            label2token[label] for label in self.labels
        ]
        self.target_token_ids: List[int] = self.tokenizer.convert_tokens_to_ids(self.target_tokens)

        # construct a mapping from label_token_id to token_ids that will be used to initialize the embeddings
        # of the labels
        self.label_embedding_weight_mapping = dict()
        for label, label_token in label2token.items():
            label_token_indices = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(label_token)
            )
            # sanity check: label_tokens should not be split up
            if len(label_token_indices) > 1:
                raise RuntimeError(f"{label_token} wrong split")
            else:
                label_token_idx = label_token_indices[0]

            label_representation = self.get_label_representation(label)
            source_indices = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(label_representation)
            )
            if self.tokenizer.unk_token_id in source_indices:
                raise RuntimeError(
                    f"tokenized label_token={label_token} [{source_indices}] contains unk_token"
                )
            self.label_embedding_weight_mapping[label_token_idx] = source_indices

    @property
    def pointer_offset(self) -> int:
        return len(self.targets)

    @property
    def target_ids(self) -> Set[int]:
        return set(range(self.pointer_offset))

    def configure_model_metric(self, stage: Optional[str] = None) -> Optional[Metric]:
        layer_metrics: dict[str, Metric] = {
            layer_name: PrecisionRecallAndF1ForLabeledAnnotations()
            for layer_name in self.layer_names
        }

        return WrappedLayerMetricsWithUnbatchAndDecodeWithErrorsFunction(
            unbatch_function=self.unbatch_output,
            decode_layers_with_errors_function=self.decode_annotations,
            layer_metrics=layer_metrics,
            error_key_correct=KEY_INVALID_CORRECT,
        )

    def reverse_relation(self, relation: Annotation) -> BinaryRelation:
        if isinstance(relation, BinaryRelation):
            reversed_label = relation.label
            if (
                reversed_label not in self.symmetric_relations
                and reversed_label != self.none_label
            ):
                reversed_label += self.REVERSED_RELATION_LABEL_SUFFIX
            reversed_rel = relation.copy(
                head=relation.tail, tail=relation.head, label=reversed_label
            )
            return reversed_rel
        else:
            raise Exception(f"reversing of relations of type {type(relation)} is not supported")

    def unreverse_relation(self, relation: Annotation) -> BinaryRelation:
        if isinstance(relation, BinaryRelation):
            head, tail, label = relation.head, relation.tail, relation.label
            # if the relation is symmetric, we sort head and tail to ensure consistent order
            if relation.label in self.symmetric_relations:
                head, tail = sorted([head, tail], key=lambda x: (x.start, x.end))
            # if the relation was reversed, we need to reconstruct the original label and swap head and tail
            elif label.endswith(self.REVERSED_RELATION_LABEL_SUFFIX):
                # reconstruct the original label and swap head and tail
                label = label[: -len(self.REVERSED_RELATION_LABEL_SUFFIX)]
                head, tail = tail, head
            return relation.copy(head=head, tail=tail, label=label)
        else:
            raise Exception(f"un-reversing of relations of type {type(relation)} is not supported")

    def encode_annotations(
        self, layers: Dict[str, Iterable[Annotation]], metadata: Optional[Dict[str, Any]] = None
    ) -> TaskOutputType:
        if not set(layers.keys()) == set(self.layer_names):
            raise Exception(f"unexpected layers: {layers.keys()}. expected: {self.layer_names}")

        if self.labels_per_layer is None:
            raise Exception("labels_per_layer is not defined. Call prepare() first or pass it in.")

        # encode relations
        all_relation_arguments = set()
        relation_arguments2label: Dict[Tuple[Annotation, ...], str] = dict()
        relation_encodings = dict()
        for rel in layers[self.relation_layer_name]:
            if not isinstance(rel, BinaryRelation):
                raise Exception(f"expected BinaryRelation, but got: {rel}")
            if rel.label in self.labels_per_layer[self.relation_layer_name]:
                if (rel.head, rel.tail) in relation_arguments2label:
                    previous_label = relation_arguments2label[(rel.head, rel.tail)]
                    if previous_label != rel.label:
                        raise ValueError(
                            f"relation {rel.head} -> {rel.tail} already exists, but has another label: "
                            f"{previous_label} (current label: {rel.label})."
                        )
                    continue
                encoded_relation = self.relation_encoder_decoder.encode(
                    annotation=rel, metadata=metadata
                )
                if encoded_relation is None:
                    raise Exception(f"failed to encode relation: {rel}")
                relation_encodings[rel] = encoded_relation
                all_relation_arguments.update([rel.head, rel.tail])
                relation_arguments2label[(rel.head, rel.tail)] = rel.label

        # encode spans that are not arguments of any relation
        no_relation_spans = [
            span for span in layers[self.span_layer_name] if span not in all_relation_arguments
        ]
        for span in no_relation_spans:
            dummy_relation = BinaryRelation(
                head=span, tail=span, label=self.loop_dummy_relation_name
            )
            encoded_relation = self.relation_encoder_decoder.encode(
                annotation=dummy_relation, metadata=metadata
            )
            if encoded_relation is not None:
                relation_encodings[dummy_relation] = encoded_relation

        # sort relations by start indices of head and tail # TODO: is this correct?
        sorted_relations = sorted(relation_encodings, key=cmp_to_key(cmp_src_rel))

        # this should never be accessed as it is, so use negative pointer offset to provoke an error
        input_len = -self.pointer_offset - 1
        if self.create_constraints:
            if metadata is None or "src_len" not in metadata:
                raise Exception("metadata with 'src_len' is required to create constraints")
            input_len = metadata["src_len"]

        # build target_ids
        target_ids = []
        constraints_list: list[Tensor] = []
        for rel in sorted_relations:
            encoded_relation = relation_encodings[rel]
            target_ids.extend(encoded_relation)

            if self.create_constraints:
                # iterate over all prefixes of the relation encoding
                for idx, t in enumerate(encoded_relation):
                    # get the constraints for the current prefix
                    current_constraints = self._build_constraint(
                        previous_ids=encoded_relation[:idx], input_len=input_len
                    )
                    # sanity check
                    if current_constraints[t] == 0:
                        raise Exception(
                            f"current_constraints[{t}] is 0, but should be 1: {current_constraints}"
                        )
                    # add the constraints to the list
                    assert isinstance(current_constraints, Tensor)
                    constraints_list.append(current_constraints)

        target_ids.append(self.eos_id)

        if self.create_constraints:
            # add constraints for the eos_id
            eos_constraint = torch.zeros(input_len + self.pointer_offset, dtype=torch.int64)
            eos_constraint[self.eos_id] = 1
            constraints_list.append(eos_constraint)
            # combine all constraints
            constraints = torch.stack(constraints_list).tolist()
        else:
            constraints = None

        # sanity check
        _, encoding_errors, remaining = self.relation_encoder_decoder.parse(encoding=target_ids)
        if (
            not all(v == 0 for k, v in encoding_errors.items() if k != "correct")
            or len(remaining) > 0
        ):
            decoded, invalid = self.decode_annotations(LabelsAndOptionalConstraints(target_ids))
            not_encoded = {}
            for layer_name in layers:
                # convert to dicts to make them comparable (original annotations are attached which breaks comparison)
                decoded_dicts = [ann.asdict() for ann in decoded[layer_name]]
                # filter annotations and convert to str to make them json serializable
                filtered = {
                    str(ann) for ann in layers[layer_name] if ann.asdict() not in decoded_dicts
                }
                if len(filtered) > 0:
                    not_encoded[layer_name] = list(filtered)
            if len(not_encoded) > 0:
                logger.warning(
                    f"encoding errors: {encoding_errors}, skipped annotations:\n"
                    f"{json.dumps(not_encoded, sort_keys=True, indent=2)}"
                )
            elif len([tag for tag in remaining if tag != self.eos_id]) > 0:
                logger.warning(
                    f"encoding errors: {encoding_errors}, remaining encoding ids: {remaining}"
                )

        return LabelsAndOptionalConstraints(labels=target_ids, constraints=constraints)

    def decode_annotations(
        self, encoding: TaskOutputType
    ) -> Tuple[Dict[str, Iterable[Annotation]], Dict[str, int]]:
        decoded_relations, errors, remaining = self.relation_encoder_decoder.parse(
            encoding=encoding.labels
        )
        relation_tuples: List[Tuple[Annotation, Annotation, str]] = []
        entity_labels: Dict[Annotation, List[str]] = defaultdict(list)
        for rel in decoded_relations:
            head_dummy = rel.head.copy(label="dummy")
            entity_labels[head_dummy].append(rel.head.label)

            if rel.label != self.loop_dummy_relation_name:
                tail_dummy = rel.tail.copy(label="dummy")
                entity_labels[tail_dummy].append(rel.tail.label)
                relation_tuples.append((head_dummy, tail_dummy, rel.label))
            else:
                assert rel.head == rel.tail

        # It may happen that some spans take part in multiple relations, but got generated with different labels.
        # In this case, we just create one span and take the most common label.
        entities: Dict[Annotation, Annotation] = {}
        for entity_dummy, labels in entity_labels.items():
            c = Counter(labels)
            # if len(c) > 1:
            #    logger.warning(f"multiple labels for span, take the most common: {dict(c)}")
            most_common_label = c.most_common(1)[0][0]
            entities[entity_dummy] = entity_dummy.copy(label=most_common_label)

        entity_layer = list(entities.values())
        relation_layer = [
            BinaryRelation(head=entities[head_dummy], tail=entities[tail_dummy], label=label)
            for head_dummy, tail_dummy, label in relation_tuples
        ]
        return {
            self.span_layer_name: entity_layer,
            self.relation_layer_name: relation_layer,
        }, errors

    def _build_constraint(
        self,
        previous_ids: List[int],
        input_len: int,
    ) -> torch.LongTensor:
        """Build a constraint for the decoder. The constraint is a binary mask that indicates which
        ids are allowed to be predicted in the next decoding step. The mask is of size input_len +
        pointer_offset, where input_len is the length of the input sequence and pointer_offset is
        the number of labels and special tokens. Uses the relation_encoder_decoder to build the
        actual constraints.

        Args:
            previous_ids: previously decoded ids
            input_len: length of the input sequence

        Returns:
            A binary mask of size input_len + pointer_offset, where 1 indicates that the id is
            allowed to be predicted next, and 0 indicates that the id is not allowed to be predicted next.
        """
        result = torch.zeros(input_len + self.pointer_offset, dtype=torch.long)
        assert isinstance(result, torch.LongTensor)
        if self.eos_id in previous_ids:
            # once eos is predicted, only allow padding
            result[self.target_pad_id] = 1
            return result

        allowed_ids, disallowed_ids = self.relation_encoder_decoder.build_decoding_constraints(
            partial_encoding=previous_ids
        )
        if allowed_ids is not None and disallowed_ids is not None:
            raise Exception(
                f"allowed_ids and disallowed_ids are both not None: {allowed_ids}, {disallowed_ids}"
            )
        elif allowed_ids is not None:
            for allowed_id in allowed_ids:
                result[allowed_id] = 1
        elif disallowed_ids is not None:
            for id in range(len(result)):
                if id not in disallowed_ids:
                    result[id] = 1
        else:
            raise Exception(
                f"allowed_ids and disallowed_ids are both None: {allowed_ids}, {disallowed_ids}"
            )
        if len(previous_ids) == 0:
            # if there are no previous ids, we also allow the eos_id
            result[self.eos_id] = 1
        else:
            # if there are previous ids, we don't allow the eos_id
            result[self.eos_id] = 0
        # never allow the bos_id
        result[self.bos_id] = 0

        return result

    def maybe_log_example(
        self,
        task_encoding: TaskEncodingType,
        targets: Optional[TargetEncodingType] = None,
    ):
        if self.log_first_n_examples is not None and self.log_first_n_examples > 0:
            tokenized_doc_id = task_encoding.metadata["tokenized_document"].id
            inputs = task_encoding.inputs
            targets = targets or task_encoding.targets
            input_tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids)
            label_tokens = [
                (
                    self.targets[target_id_or_offset]
                    if target_id_or_offset < self.pointer_offset
                    else str(target_id_or_offset)
                    + " {"
                    + str(input_tokens[target_id_or_offset - self.pointer_offset])
                    + "}"
                )
                for target_id_or_offset in targets.labels
            ]
            logger.info("*** Example ***")
            logger.info(f"doc.id:       {tokenized_doc_id}")
            logger.info(f"input_ids:    {' '.join([str(i) for i in inputs.input_ids])}")
            logger.info(f"input_tokens: {' '.join(input_tokens)}")
            logger.info(f"label_ids:    {' '.join([str(i) for i in targets.labels])}")
            logger.info(f"label_tokens: {' '.join(label_tokens)}")
            if self.create_constraints:
                # only show the shape because the content is not very readable
                logger.info(
                    f"constraints:  {torch.tensor(targets.constraints).shape} (content is omitted)"
                )
            self.log_first_n_examples -= 1

    def tokenize_document(self, document: DocumentType) -> List[TokenBasedDocument]:
        field_mapping = dict(self.annotation_field_mapping)
        if self.partition_layer_name is not None:
            field_mapping[self.partition_layer_name] = "labeled_partitions"
            partition_layer = "labeled_partitions"
        else:
            partition_layer = None
        casted_document = document.as_type(self.document_type, field_mapping=field_mapping)
        tokenized_docs = tokenize_document(
            casted_document,
            tokenizer=self.tokenizer,
            result_document_type=self.tokenized_document_type,
            partition_layer=partition_layer,
            **self.tokenizer_kwargs,
        )
        for idx, tokenized_doc in enumerate(tokenized_docs):
            tokenized_doc.id = f"{document.id}-tokenized-{idx+1}-of-{len(tokenized_docs)}"

        return tokenized_docs

    def encode_input(
        self, document: DocumentType, is_training: bool = False
    ) -> Optional[Union[TaskEncodingType, Sequence[TaskEncodingType]]]:
        tokenized_docs = self.tokenize_document(document)
        task_encodings: List[TaskEncodingType] = []
        for tokenized_doc in tokenized_docs:
            tokenizer_encoding = tokenized_doc.metadata["tokenizer_encoding"]
            task_encodings.append(
                TaskEncoding(
                    document=document,
                    inputs=InputEncodingType(
                        input_ids=tokenizer_encoding.ids,
                        attention_mask=tokenizer_encoding.attention_mask,
                    ),
                    metadata={"tokenized_document": tokenized_doc},
                )
            )

        return task_encodings

    def get_mapped_layer(self, document: Document, layer_name: str) -> AnnotationLayer:
        if layer_name in self.annotation_field_mapping:
            layer_name = self.annotation_field_mapping[layer_name]
        return document[layer_name]

    def encode_target(self, task_encoding: TaskEncodingType) -> Optional[TargetEncodingType]:
        try:
            document = task_encoding.metadata["tokenized_document"]

            layers = {
                layer_name: self.get_mapped_layer(document, layer_name=layer_name)
                for layer_name in self.layer_names
            }

            if self.add_reversed_relations:
                # create a copy to avoid modifying the annotation layer in the document
                relations = list(layers[self.relation_layer_name])
                reversed_relations = [self.reverse_relation(rel) for rel in relations]
                layers[self.relation_layer_name] = relations + reversed_relations

            result = self.encode_annotations(
                layers=layers,
                metadata={
                    **task_encoding.metadata,
                    "src_len": len(task_encoding.inputs.input_ids),
                },
            )

            self.maybe_log_example(task_encoding=task_encoding, targets=result)
            return result
        except Exception as e:
            logger.error(f"failed to encode target, it will be skipped: {e}")
            return None

    def collate(self, task_encodings: Sequence[TaskEncodingType]) -> TaskBatchEncoding:
        if len(task_encodings) == 0:
            raise ValueError("no task_encodings available")
        inputs = InputEncodingType.batch(
            values=[x.inputs for x in task_encodings],
            dtypes=self.dtypes,
            pad_values=self.pad_values,
        )

        targets = None
        if task_encodings[0].has_targets:
            targets = TargetEncodingType.batch(
                values=[x.targets for x in task_encodings],
                dtypes=self.dtypes,
                pad_values=self.pad_values,
            )

        return inputs, targets

    def unbatch_output(self, model_output: ModelBatchOutput) -> Sequence[TaskOutputType]:
        labels = model_output["labels"]
        batch_size = labels.size(0)

        # We use the position after the first eos token as the seq_len.
        # Note that, if eos_id is not in model_output for a given batch item, the result will be
        # model_output.size(1) + 1 (i.e. seq_len + 1) for that batch item. This is fine, because we use the
        # seq_lengths just to truncate the output and want to keep everything if eos_id is not present.
        seq_lengths = get_first_occurrence_index(labels, self.eos_id) + 1

        result = [
            LabelsAndOptionalConstraints(labels[i, : seq_lengths[i]].to(device="cpu").tolist())
            for i in range(batch_size)
        ]
        return result

    def create_annotations_from_output(
        self,
        task_encoding: TaskEncodingType,
        task_output: TaskOutputType,
    ) -> Iterator[Tuple[str, Annotation]]:
        layers, errors = self.decode_annotations(
            encoding=task_output,  # metadata=task_encoding.metadata
        )
        tokenized_document = task_encoding.metadata["tokenized_document"]

        # Note: token_based_document_to_text_based() does not yet consider predictions, so we need to clear
        # the main annotations and attach the predictions to that
        for layer_name, annotations in layers.items():
            layer = self.get_mapped_layer(tokenized_document, layer_name=layer_name)
            layer.clear()
            layer.extend(annotations)

        untokenized_document = token_based_document_to_text_based(
            tokenized_document, result_document_type=self.document_type
        )

        for layer_name in layers:
            annotations = self.get_mapped_layer(untokenized_document, layer_name=layer_name)
            for annotation in annotations:
                # handle relations that may be reversed
                if layer_name == self.relation_layer_name and self.add_reversed_relations:
                    unreversed_relation = self.unreverse_relation(annotation)
                    yield layer_name, unreversed_relation
                else:
                    yield layer_name, annotation.copy()
