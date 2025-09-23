"""
workflow:
    Document
        -> (InputEncoding, TargetEncoding) -> TaskEncoding -> TaskBatchEncoding
            -> ModelBatchEncoding -> ModelBatchOutput
        -> TaskOutput
    -> Document
"""

import logging
from collections import defaultdict
from functools import partial
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypedDict,
    Union,
)

import numpy as np
import torch
from pie_core import Annotation, AnnotationLayer, Document, TaskEncoding, TaskModule
from pie_documents.annotations import (
    BinaryRelation,
    LabeledSpan,
    MultiLabeledBinaryRelation,
    NaryRelation,
    Span,
)
from pie_documents.documents import (
    TextBasedDocument,
    TextDocumentWithLabeledSpansAndBinaryRelations,
    TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
)
from pie_documents.utils.span import distance as span_distance
from pie_documents.utils.span import is_contained_in
from torch import LongTensor
from torchmetrics import ClasswiseWrapper, F1Score, MetricCollection
from transformers import AutoTokenizer
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy
from typing_extensions import TypeAlias, TypeVar

from pytorch_ie.models.simple_sequence_classification import InputType as ModelInputType
from pytorch_ie.models.simple_sequence_classification import TargetType as ModelTargetType
from pytorch_ie.taskmodules.interface import ChangesTokenizerVocabSize
from pytorch_ie.utils.window import get_window_around_slice

from ..utils.document import SpanNotAlignedWithTokenException, get_aligned_token_span
from .common.mixins import RelationStatisticsMixin
from .metrics import WrappedMetricWithPrepareFunction

# from pytorch_ie.utils.tokenization import (
#    SpanNotAlignedWithTokenException,
#    get_aligned_token_span,
# )

InputEncodingType: TypeAlias = Dict[str, Any]
TargetEncodingType: TypeAlias = Sequence[int]
DocumentType: TypeAlias = TextBasedDocument

TaskEncodingType: TypeAlias = TaskEncoding[
    DocumentType,
    InputEncodingType,
    TargetEncodingType,
]


class TaskOutputType(TypedDict, total=False):
    labels: Sequence[str]
    probabilities: Sequence[float]


TaskModuleType: TypeAlias = TaskModule[
    # _InputEncoding, _TargetEncoding, _TaskBatchEncoding, _ModelBatchOutput, _TaskOutput
    DocumentType,
    InputEncodingType,
    TargetEncodingType,
    Tuple[ModelInputType, Optional[ModelTargetType]],
    ModelTargetType,
    TaskOutputType,
]


HEAD = "head"
TAIL = "tail"
START = "start"
END = "end"


logger = logging.getLogger(__name__)


def _get_labels(model_output: ModelTargetType) -> LongTensor:
    result = model_output["labels"]
    assert isinstance(result, LongTensor)
    return result


def _get_labels_together_remove_none_label(
    predictions: ModelTargetType, targets: ModelTargetType, none_idx: int
) -> Tuple[LongTensor, LongTensor]:
    mask_not_both_none = (predictions["labels"] != none_idx) | (targets["labels"] != none_idx)
    predictions_not_none = predictions["labels"][mask_not_both_none]
    targets_not_none = targets["labels"][mask_not_both_none]
    assert isinstance(predictions_not_none, LongTensor)
    assert isinstance(targets_not_none, LongTensor)
    return predictions_not_none, targets_not_none


def find_sublist(sub: List, bigger: List) -> int:
    if not bigger:
        return -1
    if not sub:
        return 0
    first, rest = sub[0], sub[1:]
    pos = 0
    try:
        while True:
            pos = bigger.index(first, pos) + 1
            if not rest or bigger[pos : pos + len(rest)] == rest:
                return pos - 1
    except ValueError:
        return -1


class MarkerFactory:
    def __init__(self, role_to_marker: Dict[str, str]):
        self.role_to_marker = role_to_marker

    def _get_role_marker(self, role: str) -> str:
        return self.role_to_marker[role]

    def _get_marker(self, role: str, is_start: bool, label: Optional[str] = None) -> str:
        result = "["
        if not is_start:
            result += "/"
        result += self._get_role_marker(role)
        if label is not None:
            result += f":{label}"
        result += "]"
        return result

    def get_start_marker(self, role: str, label: Optional[str] = None) -> str:
        return self._get_marker(role=role, is_start=True, label=label)

    def get_end_marker(self, role: str, label: Optional[str] = None) -> str:
        return self._get_marker(role=role, is_start=False, label=label)

    def get_append_marker(self, role: str, label: Optional[str] = None) -> str:
        role_marker = self._get_role_marker(role)
        if label is None:
            return f"[{role_marker}]"
        else:
            return f"[{role_marker}={label}]"

    @property
    def all_roles(self) -> Set[str]:
        return set(self.role_to_marker)

    def get_all_markers(
        self,
        entity_labels: List[str],
        append_markers: bool = False,
        add_type_to_marker: bool = False,
    ) -> List[str]:
        result: Set[str] = set()
        if add_type_to_marker:
            none_and_labels = [None] + entity_labels
        else:
            none_and_labels = [None]
        for role in self.all_roles:
            # create start and end markers without label and for all labels, if add_type_to_marker
            for maybe_label in none_and_labels:
                result.add(self.get_start_marker(role=role, label=maybe_label))
                result.add(self.get_end_marker(role=role, label=maybe_label))
            # create append markers for all labels
            if append_markers:
                for entity_label in entity_labels:
                    result.add(self.get_append_marker(role=role, label=entity_label))

        # sort and convert to list
        return sorted(result)


class RelationArgument:
    def __init__(
        self,
        entity: LabeledSpan,
        role: str,
        token_span: Span,
        add_type_to_marker: bool,
        marker_factory: MarkerFactory,
    ) -> None:
        self.marker_factory = marker_factory
        if role not in self.marker_factory.all_roles:
            raise ValueError(
                f"role='{role}' not in known roles={sorted(self.marker_factory.all_roles)} (did you "
                f"initialise the taskmodule with the correct argument_role_to_marker dictionary?)"
            )

        self.entity = entity

        self.role = role
        self.token_span = token_span
        self.add_type_to_marker = add_type_to_marker

    @property
    def maybe_label(self) -> Optional[str]:
        return self.entity.label if self.add_type_to_marker else None

    @property
    def as_start_marker(self) -> str:
        return self.marker_factory.get_start_marker(role=self.role, label=self.maybe_label)

    @property
    def as_end_marker(self) -> str:
        return self.marker_factory.get_end_marker(role=self.role, label=self.maybe_label)

    @property
    def as_append_marker(self) -> str:
        # Note: we add the label in either case (we use self.entity.label instead of self.label)
        return self.marker_factory.get_append_marker(role=self.role, label=self.entity.label)

    def shift_token_span(self, value: int):
        self.token_span = Span(
            start=self.token_span.start + value, end=self.token_span.end + value
        )


def get_relation_argument_spans_and_roles(
    relation: Annotation,
) -> Tuple[Tuple[str, Annotation], ...]:
    if isinstance(relation, BinaryRelation):
        return (HEAD, relation.head), (TAIL, relation.tail)
    elif isinstance(relation, NaryRelation):
        # create unique order by sorting the arguments by their start and end positions and role
        sorted_args = sorted(
            zip(relation.roles, relation.arguments),
            key=lambda role_and_span: (
                role_and_span[1].start,
                role_and_span[1].end,
                role_and_span[0],
            ),
        )
        return tuple(sorted_args)
    else:
        raise NotImplementedError(
            f"the taskmodule does not yet support getting relation arguments for type: {type(relation)}"
        )


def construct_mask(input_ids: torch.LongTensor, positive_ids: List[Any]) -> torch.LongTensor:
    """Construct a mask for the input_ids where all entries in mask_ids are 1."""
    masks = [torch.nonzero(input_ids == marker_token_id) for marker_token_id in positive_ids]
    global_mask = torch.cat(masks)
    value = torch.ones(global_mask.shape[0], dtype=torch.long)
    mask = torch.zeros(input_ids.shape, dtype=torch.long)
    mask.index_put_(tuple(global_mask.t()), value)
    assert isinstance(mask, torch.LongTensor)
    return mask


S = TypeVar("S", bound=Span)


def shift_span(span: S, offset: int) -> S:
    return span.copy(start=span.start + offset, end=span.end + offset)


def bio_encode_spans(
    spans: List[Tuple[int, int, str]], total_length: int, label2idx: Dict[str, int]
) -> List[int]:
    # result = ["O"] * total_length
    result = [0] * total_length
    for start, end, label in spans:
        # result[start] = f"B-{label}"
        result[start] = label2idx[label] * 2 + 1
        for i in range(start + 1, end):
            # result[i] = f"I-{label}"
            result[i] = label2idx[label] * 2 + 2
    return result


@TaskModule.register()
class RETextClassificationWithIndicesTaskModule(
    RelationStatisticsMixin,
    TaskModuleType,
    ChangesTokenizerVocabSize,
):
    """Marker based relation extraction. This taskmodule prepares the input token ids in such a way
    that before and after the candidate head and tail entities special marker tokens are inserted.
    Then, the modified token ids can be simply passed into a transformer based text classifier
    model.

    parameters:

        partition_annotation: str, optional. If specified, LabeledSpan annotations with this name are
            expected to define partitions of the document that will be processed individually, e.g. sentences
            or sections of the document text.
        none_label: str, defaults to "no_relation". The relation label that indicate dummy/negative relations.
            Predicted relations with that label will not be added to the document(s).
        max_window: int, optional. If specified, use the tokens in a window of maximal this amount of tokens
            around the center of head and tail entities and pass only that into the transformer.
        create_relation_candidates: bool, defaults to False. If True, create relation candidates by pairwise
            combining all entities in the document and assigning the none_label. If the document already contains
            a relation with the entity pair, we do not add it again. If False, assume that the document already
            contains relation annotations including negative examples (i.e. relations with the none_label).
        handle_relations_with_same_arguments: str, defaults to "keep_none". If "keep_none", all relations that
            share same arguments will be removed. If "keep_first", first occurred duplicate will be kept.
        argument_type_whitelist: List[List[str]], optional, defaults to None. If set, only relations (candidates)
            with given argument type tuples are created from document and by by `create_relation_candidates`.
            This affects only model input.
        argument_and_relation_type_whitelist: Union[Dict[str, List[List[str]]], List[List[str]]], optional,
            defaults None. If set, only given relation types with given argument types will persist in
            documents and generated by `create_relation_candidates`. This also affects predictions on
            `decode()`, so it strictly filters both model input and output. Can also be passed as a list
            of lists, where the first element is the relation type and the rest are the argument types.
    """

    PREPARED_ATTRIBUTES = ["labels", "entity_labels"]

    def __init__(
        self,
        tokenizer_name_or_path: str,
        relation_annotation: str = "binary_relations",
        add_candidate_relations: bool = False,
        add_reversed_relations: bool = False,
        partition_annotation: Optional[str] = None,
        none_label: str = "no_relation",
        padding: Union[bool, str, PaddingStrategy] = True,
        truncation: Union[bool, str, TruncationStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        multi_label: bool = False,
        labels: Optional[List[str]] = None,
        label_to_id: Optional[Dict[str, int]] = None,
        add_type_to_marker: bool = False,
        argument_role_to_marker: Optional[Dict[str, str]] = None,
        single_argument_pair: bool = True,
        append_markers: bool = False,
        insert_markers: bool = True,
        entity_labels: Optional[List[str]] = None,
        reversed_relation_label_suffix: str = "_reversed",
        symmetric_relations: Optional[List[str]] = None,
        reverse_symmetric_relations: bool = True,
        max_argument_distance: Optional[int] = None,
        max_argument_distance_type: str = "inner",
        max_argument_distance_tokens: Optional[int] = None,
        max_argument_distance_type_tokens: str = "inner",
        max_window: Optional[int] = None,
        allow_discontinuous_text: bool = False,
        log_first_n_examples: int = 0,
        add_argument_indices_to_input: bool = False,
        add_argument_tags_to_input: bool = False,
        add_entity_tags_to_input: bool = False,
        add_global_attention_mask_to_input: bool = False,
        argument_type_whitelist: Optional[List[List[str]]] = None,
        handle_relations_with_same_arguments: str = "keep_none",
        argument_and_relation_type_whitelist: Optional[
            Union[Dict[str, List[List[str]]], List[List[str]]]
        ] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if label_to_id is not None:
            logger.warning(
                "The parameter label_to_id is deprecated and will be removed in a future version. "
                "Please use labels instead."
            )
            id_to_label = {v: k for k, v in label_to_id.items()}
            # reconstruct labels from label_to_id. Note that we need to remove the none_label
            labels = [
                id_to_label[i] for i in range(len(id_to_label)) if id_to_label[i] != none_label
            ]
        self.save_hyperparameters(ignore=["label_to_id"])

        self.relation_annotation = relation_annotation
        self.add_candidate_relations = add_candidate_relations
        self.add_reversed_relations = add_reversed_relations
        self.padding = padding
        self.truncation = truncation
        self.labels = labels
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.multi_label = multi_label
        self.add_type_to_marker = add_type_to_marker
        self.single_argument_pair = single_argument_pair
        self.append_markers = append_markers
        self.insert_markers = insert_markers
        self.entity_labels = entity_labels
        self.partition_annotation = partition_annotation
        self.none_label = none_label
        self.reversed_relation_label_suffix = reversed_relation_label_suffix
        self.symmetric_relations = set(symmetric_relations or [])
        self.reverse_symmetric_relations = reverse_symmetric_relations
        self.max_argument_distance = max_argument_distance
        self.max_argument_distance_type = max_argument_distance_type
        self.max_argument_distance_tokens = max_argument_distance_tokens
        self.max_argument_distance_type_tokens = max_argument_distance_type_tokens
        self.max_window = max_window
        self.allow_discontinuous_text = allow_discontinuous_text
        self.handle_relations_with_same_arguments = handle_relations_with_same_arguments
        self.argument_type_whitelist: Optional[Set[Tuple[str, ...]]] = None
        self.argument_and_relation_type_whitelist: Optional[Dict[str, Set[Tuple[str, ...]]]] = None

        if argument_type_whitelist is not None:
            # hydra does not support tuples, so we got lists and need to convert them
            self.argument_type_whitelist = {tuple(types) for types in argument_type_whitelist}
        if argument_and_relation_type_whitelist is not None:
            # hydra does not support tuples, so we got lists and need to convert them
            if isinstance(argument_and_relation_type_whitelist, list):
                self.argument_and_relation_type_whitelist = defaultdict(set)
                for types_list in argument_and_relation_type_whitelist:
                    if len(types_list) < 1:
                        raise ValueError(
                            "argument_and_relation_type_whitelist must be a list of lists with at least one element"
                        )
                    self.argument_and_relation_type_whitelist[types_list[0]].add(
                        tuple(types_list[1:])
                    )
            else:
                self.argument_and_relation_type_whitelist = {
                    rel: {tuple(types) for types in types_list}
                    for rel, types_list in argument_and_relation_type_whitelist.items()
                }
        # overwrite None with 0 for backward compatibility
        self.log_first_n_examples = log_first_n_examples or 0
        self.add_argument_indices_to_input = add_argument_indices_to_input
        self.add_argument_tags_to_input = add_argument_tags_to_input
        self.add_entity_tags_to_input = add_entity_tags_to_input
        self.add_global_attention_mask_to_input = add_global_attention_mask_to_input
        if argument_role_to_marker is None:
            self.argument_role_to_marker = {HEAD: "H", TAIL: "T"}
        else:
            self.argument_role_to_marker = argument_role_to_marker

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

        # used when allow_discontinuous_text
        self.glue_token_ids = self._get_glue_token_ids()

        self.argument_markers = None

        self._logged_examples_counter = 0

    def _get_glue_token_ids(self):
        dummy_ids = self.tokenizer.build_inputs_with_special_tokens(
            token_ids_0=[-1], token_ids_1=[-2]
        )
        return dummy_ids[dummy_ids.index(-1) + 1 : dummy_ids.index(-2)]

    @property
    def document_type(self) -> Optional[Type[DocumentType]]:
        if self.partition_annotation is not None:
            dt = TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions
        else:
            dt = TextDocumentWithLabeledSpansAndBinaryRelations
        if self.relation_annotation == "binary_relations":
            return dt
        else:
            logger.warning(
                f"relation_annotation={self.relation_annotation} is "
                f"not the default value ('binary_relations'), so the taskmodule {type(self).__name__} can not request "
                f"the usual document type for auto-conversion ({dt.__name__}) because this has the bespoken default "
                f"value as layer name instead of the provided one."
            )
            return None

    def get_relation_layer(self, document: Document) -> AnnotationLayer[BinaryRelation]:
        return document[self.relation_annotation]

    def get_entity_layer(self, document: Document) -> AnnotationLayer[LabeledSpan]:
        relations: AnnotationLayer[BinaryRelation] = self.get_relation_layer(document)
        return relations.target_layer

    def get_marker_factory(self) -> MarkerFactory:
        return MarkerFactory(role_to_marker=self.argument_role_to_marker)

    def _prepare(self, documents: Sequence[DocumentType]) -> None:
        entity_labels: Set[str] = set()
        relation_labels: Set[str] = set()
        for document in documents:
            relations: AnnotationLayer[BinaryRelation] = self.get_relation_layer(document)
            entities: AnnotationLayer[LabeledSpan] = self.get_entity_layer(document)

            for entity in entities:
                entity_labels.add(entity.label)

            for relation in relations:
                relation_labels.add(relation.label)
                if self.add_reversed_relations:
                    if relation.label.endswith(self.reversed_relation_label_suffix):
                        raise ValueError(
                            f"doc.id={document.id}: the relation label '{relation.label}' already ends with "
                            f"the reversed_relation_label_suffix '{self.reversed_relation_label_suffix}', "
                            f"this is not allowed because we would not know if we should strip the suffix and "
                            f"revert the arguments during inference or not"
                        )
                    if relation.label not in self.symmetric_relations:
                        relation_labels.add(relation.label + self.reversed_relation_label_suffix)

        if self.none_label in relation_labels:
            relation_labels.remove(self.none_label)

        self.labels = sorted(relation_labels)
        self.entity_labels = sorted(entity_labels)

    def encode(self, *args, **kwargs):
        self.reset_statistics()
        res = super().encode(*args, **kwargs)
        self.show_statistics()
        return res

    def _post_prepare(self):
        self.label_to_id = {label: i + 1 for i, label in enumerate(self.labels)}
        self.label_to_id[self.none_label] = 0
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

        self.marker_factory = self.get_marker_factory()
        self.argument_markers = self.marker_factory.get_all_markers(
            append_markers=self.append_markers,
            add_type_to_marker=self.add_type_to_marker,
            entity_labels=self.entity_labels,
        )
        self.tokenizer.add_tokens(self.argument_markers, special_tokens=True)

        self.argument_markers_to_id = {
            marker: self.tokenizer.vocab[marker] for marker in self.argument_markers
        }

        self.argument_role2idx = {
            role: i for i, role in enumerate(sorted(self.marker_factory.all_roles))
        }

    def _add_reversed_relations(
        self,
        arguments2relation: Dict[Tuple[Tuple[str, Annotation], ...], Annotation],
        doc_id: Optional[str] = None,
    ) -> None:
        if self.add_reversed_relations:
            for arguments, rel in list(arguments2relation.items()):
                arg_roles, arg_spans = zip(*arguments)
                if isinstance(rel, BinaryRelation):
                    label = rel.label
                    if label in self.symmetric_relations and not self.reverse_symmetric_relations:
                        continue
                    if label.endswith(self.reversed_relation_label_suffix):
                        raise ValueError(
                            f"doc.id={doc_id}: The relation has the label '{label}' which already ends with the "
                            f"reversed_relation_label_suffix='{self.reversed_relation_label_suffix}'. "
                            f"It looks like the relation is already reversed, which is not allowed."
                        )
                    if rel.label not in self.symmetric_relations:
                        label += self.reversed_relation_label_suffix

                    reversed_rel = BinaryRelation(
                        head=rel.tail,
                        tail=rel.head,
                        label=label,
                        score=rel.score,
                    )
                    reversed_arguments = get_relation_argument_spans_and_roles(reversed_rel)
                    if reversed_arguments in arguments2relation:
                        prev_rel = arguments2relation[reversed_arguments]
                        prev_label = prev_rel.label
                        logger.warning(
                            f"doc.id={doc_id}: there is already a relation with reversed "
                            f"arguments={reversed_arguments} and label={prev_label}, so we do not add the reversed "
                            f"relation (with label {prev_label}) for these arguments"
                        )
                        if self.collect_statistics:
                            self.collect_relation("skipped_reversed_same_arguments", reversed_rel)
                        continue
                    elif rel.label in self.symmetric_relations:
                        # warn if the original relation arguments were not sorted by their start and end positions
                        # in the case of symmetric relations
                        if not all(isinstance(arg_span, Span) for arg_span in arg_spans):
                            raise NotImplementedError(
                                f"doc.id={doc_id}: the taskmodule does not yet support adding reversed relations "
                                f"for symmetric relations with arguments that are no Spans: {arguments}"
                            )
                        args_sorted = sorted(
                            [rel.head, rel.tail], key=lambda span: (span.start, span.end)
                        )
                        if args_sorted != [rel.head, rel.tail]:
                            logger.warning(
                                f"doc.id={doc_id}: The symmetric relation with label '{label}' has arguments "
                                f"{arguments} which are not sorted by their start and end positions. "
                                f"This may lead to problems during evaluation because we assume that the "
                                f"arguments of symmetric relations were sorted in the beginning and, thus, interpret "
                                f"relations where this is not the case as reversed. All reversed relations will get "
                                f"their arguments swapped during inference in the case of add_reversed_relations=True "
                                f"to remove duplicates. You may consider adding reversed versions of the *symmetric* "
                                f"relations on your own and then setting *reverse_symmetric_relations* to False."
                            )
                            if self.collect_statistics:
                                self.collect_relation(
                                    "used_not_sorted_reversed_arguments", reversed_rel
                                )

                    arguments2relation[reversed_arguments] = reversed_rel
                else:
                    raise NotImplementedError(
                        f"doc.id={doc_id}: the taskmodule does not yet support adding reversed relations for type: "
                        f"{type(rel)}"
                    )

    def _filter_relations_by_argument_and_relation_type_whitelist(
        self,
        arguments2relation: Dict[Tuple[Tuple[str, Annotation], ...], Annotation],
        doc_id: Optional[str] = None,
    ) -> None:
        if self.argument_and_relation_type_whitelist is not None:
            for arguments, relation in list(arguments2relation.items()):
                argument_labels = tuple(getattr(ann, "label") for role, ann in arguments)
                relation_label = getattr(relation, "label")
                if (
                    relation_label not in self.argument_and_relation_type_whitelist
                    or argument_labels
                    not in self.argument_and_relation_type_whitelist[relation_label]
                ):
                    rel = arguments2relation.pop(arguments)
                    self.collect_relation("skipped_argument_and_relation_type_whitelist", rel)

    def _filter_relations_by_argument_type_whitelist(
        self,
        arguments2relation: Dict[Tuple[Tuple[str, Annotation], ...], Annotation],
        doc_id: Optional[str] = None,
    ) -> None:
        if self.argument_type_whitelist is not None:
            for arguments, rel in list(arguments2relation.items()):
                argument_labels = tuple(getattr(arg, "label") for _, arg in arguments)
                if argument_labels not in self.argument_type_whitelist:
                    rel = arguments2relation.pop(arguments)
                    self.collect_relation("skipped_argument_type_whitelist", rel)

    def _add_candidate_relations(
        self,
        arguments2relation: Dict[Tuple[Tuple[str, Annotation], ...], Annotation],
        entities: Iterable[Span],
        arguments_blacklist: Optional[Set[Tuple[Tuple[str, Annotation], ...]]] = None,
        doc_id: Optional[str] = None,
    ) -> None:
        if self.add_candidate_relations:
            if self.marker_factory.all_roles == {HEAD, TAIL}:
                # flatten argument_and_relation_type_whitelist values
                arg_rel_whitelist_vals_set = (
                    None
                    if self.argument_and_relation_type_whitelist is None
                    else {i for j in self.argument_and_relation_type_whitelist.values() for i in j}
                )
                # iterate over all possible argument candidates
                for head in entities:
                    for tail in entities:
                        if head == tail:
                            continue

                        # Create a relation candidate with the none label. Otherwise, we use the existing relation.
                        new_relation = BinaryRelation(
                            head=head, tail=tail, label=self.none_label, score=1.0
                        )
                        new_relation_args = get_relation_argument_spans_and_roles(new_relation)
                        arg_roles, arg_spans = zip(*new_relation_args)
                        arg_labels = tuple(getattr(ann, "label") for ann in arg_spans)

                        # Skip if argument_type_whitelist and/or argument_and_relation_type_whitelist
                        # are defined and current candidates do not fit.
                        if (
                            self.argument_type_whitelist is not None
                            and arg_labels not in self.argument_type_whitelist
                        ) or (
                            arg_rel_whitelist_vals_set is not None
                            and arg_labels not in arg_rel_whitelist_vals_set
                        ):
                            continue

                        # check blacklist
                        if (
                            arguments_blacklist is not None
                            and new_relation_args in arguments_blacklist
                        ):
                            continue

                        # we use the new relation only if there is no existing relation with the same arguments
                        if new_relation_args not in arguments2relation:
                            arguments2relation[new_relation_args] = new_relation
            else:
                raise NotImplementedError(
                    f"doc.id={doc_id}: the taskmodule does not yet support adding relation candidates "
                    f"with argument roles other than 'head' and 'tail': {sorted(self.marker_factory.all_roles)}"
                )

    def _filter_relations_by_argument_distance(
        self,
        arguments2relation: Dict[Tuple[Tuple[str, Annotation], ...], Annotation],
        doc_id: Optional[str] = None,
    ) -> None:
        if self.max_argument_distance is not None:
            for arguments, rel in list(arguments2relation.items()):
                if isinstance(rel, BinaryRelation):
                    if isinstance(rel.head, Span) and isinstance(rel.tail, Span):
                        dist = span_distance(
                            (rel.head.start, rel.head.end),
                            (rel.tail.start, rel.tail.end),
                            self.max_argument_distance_type,
                        )
                        if dist > self.max_argument_distance:
                            arguments2relation.pop(arguments)
                            self.collect_relation("skipped_argument_distance", rel)
                    else:
                        raise NotImplementedError(
                            f"doc.id={doc_id}: the taskmodule does not yet support filtering relation candidates "
                            f"with arguments of type: {type(rel.head)} and {type(rel.tail)}"
                        )
                else:
                    raise NotImplementedError(
                        f"doc.id={doc_id}: the taskmodule does not yet support filtering relation candidates for "
                        f"type: {type(rel)}"
                    )

    def encode_input(
        self,
        document: DocumentType,
        is_training: bool = False,
    ) -> Optional[Union[TaskEncodingType, Sequence[TaskEncodingType]]]:
        all_relations: Sequence[Annotation] = self.get_relation_layer(document)
        all_entities: Sequence[Span] = self.get_entity_layer(document)
        self.collect_all_relations("available", all_relations)

        partitions: Sequence[Span]
        if self.partition_annotation is not None:
            partitions = document[self.partition_annotation]
            if len(partitions) == 0:
                logger.warning(
                    f"the document {document.id} has no '{self.partition_annotation}' partition entries, "
                    f"no inputs will be created!"
                )
        else:
            # use single dummy partition
            partitions = [Span(start=0, end=len(document.text))]

        task_encodings: List[TaskEncodingType] = []
        for partition in partitions:
            # get all entities that are contained in the current partition
            entities: List[Span] = [
                entity
                for entity in all_entities
                if is_contained_in((entity.start, entity.end), (partition.start, partition.end))
            ]

            # Create a mapping from relation arguments to the respective relation objects.
            # Note that the data can contain multiple relations with the same arguments.
            entities_set = set(entities)
            arguments2relations: Dict[Tuple[Tuple[str, Annotation], ...], List[Annotation]] = (
                defaultdict(list)
            )
            for rel in all_relations:
                # Skip relations with unknown labels. Use label_to_id because that contains the none_label
                if rel.label not in self.label_to_id:
                    self.collect_relation("skipped_unknown_label", rel)
                    continue

                arguments = get_relation_argument_spans_and_roles(rel)
                arg_roles, arg_spans = zip(*arguments)

                # filter out all relations that are completely outside the current partition
                if all(arg_span not in entities_set for arg_span in arg_spans):
                    continue

                # filter relations that are only partially contained in the current partition,
                # i.e. some arguments are in the partition and some are not
                if any(arg_span not in entities_set for arg_span in arg_spans):
                    logger.warning(
                        f"doc.id={document.id}: there is a relation with label '{rel.label}' and arguments "
                        f"{arguments} that is only partially contained in the current partition. "
                        f"We skip this relation."
                    )
                    self.collect_relation("skipped_partially_contained", rel)
                    continue
                arguments2relations[arguments].append(rel)

            # resolve duplicates for same arguments
            arguments2relation: Dict[Tuple[Tuple[str, Annotation], ...], Annotation] = {}
            # we will never create an encoding for the relation candidates in arguments_blacklist
            arguments_blacklist: Set[Tuple[Tuple[str, Annotation], ...]] = set()
            for arguments, relations in arguments2relations.items():
                relations_set = set(relations)
                # more than one unique relation with the same arguments
                if len(relations_set) > 1:
                    arguments_resolved = tuple(map(lambda x: (x[0], x[1].resolve()), arguments))
                    labels = [rel.label for rel in relations]
                    if self.handle_relations_with_same_arguments == "keep_first":
                        # keep only the first relation
                        arguments2relation[arguments] = relations[0]
                        for discard_rel in set(relations) - {
                            relations[0]
                        }:  # remove all other relations
                            self.collect_relation("skipped_same_arguments", discard_rel)
                        if not self.collect_statistics:
                            # We show this warning only if statistics are disabled.
                            # We want to be informed if such skip occurs, but having it in statistics and
                            # getting lots of warnings in the same time seemed overwhelming.
                            logger.warning(
                                f"doc.id={document.id}: there are multiple relations with the same arguments "
                                f"{arguments_resolved}, but different labels: {labels}. We only keep the first "
                                f"occurring relation which has the label='{relations[0].label}'."
                            )
                    elif self.handle_relations_with_same_arguments == "keep_none":
                        # add these arguments to the blacklist to not add them as 'no-relation's back again
                        arguments_blacklist.add(arguments)
                        # remove all relations with the same arguments
                        for discard_rel in relations_set:
                            self.collect_relation("skipped_same_arguments", discard_rel)
                        if not self.collect_statistics:
                            logger.warning(
                                f"doc.id={document.id}: there are multiple relations with the same arguments "
                                f"{arguments_resolved}, but different labels: {labels}. All relations will be removed."
                            )
                    else:
                        raise ValueError(
                            f"'handle_relations_with_same_arguments' must be 'keep_first' or 'keep_none', "
                            f"but got `{self.handle_relations_with_same_arguments}`."
                        )
                else:
                    arguments2relation[arguments] = relations[0]
                    # more than one duplicate relation (with the same arguments)
                    if len(relations) > 1:
                        # if 'collect_statistics=true' such duplicates won't be collected and are not counted in
                        # statistics if 'collect_statistics=true' either as 'available' or as 'skipped_same_arguments'
                        logger.warning(
                            f"doc.id={document.id}: Relation annotation `{rel.resolve()}` is duplicated. "
                            f"We keep only one of them. Duplicate won't appear in statistics either as 'available' "
                            f"or as skipped."
                        )

            # We use this filter before adding reversed relations because we also don't want them to be reversed
            self._filter_relations_by_argument_and_relation_type_whitelist(
                arguments2relation=arguments2relation, doc_id=document.id
            )
            self._add_reversed_relations(arguments2relation=arguments2relation, doc_id=document.id)
            self._filter_relations_by_argument_type_whitelist(
                arguments2relation=arguments2relation, doc_id=document.id
            )
            self._add_candidate_relations(
                arguments2relation=arguments2relation,
                arguments_blacklist=arguments_blacklist,
                entities=entities,
                doc_id=document.id,
            )

            self._filter_relations_by_argument_distance(
                arguments2relation=arguments2relation, doc_id=document.id
            )

            without_special_tokens = self.max_window is not None
            text = document.text[partition.start : partition.end]
            encoding = self.tokenizer(
                text,
                padding=False,
                truncation=self.truncation if self.max_window is None else False,
                max_length=self.max_length,
                is_split_into_words=False,
                return_offsets_mapping=False,
                add_special_tokens=not without_special_tokens,
            )

            for arguments, rel in arguments2relation.items():
                arg_roles, arg_spans = zip(*arguments)
                if not all(isinstance(arg, LabeledSpan) for arg in arg_spans):
                    # TODO: add test case for this
                    raise ValueError(
                        f"the taskmodule expects the relation arguments to be of type LabeledSpan, "
                        f"but got {[type(arg) for arg in arg_spans]}"
                    )

                arg_spans_partition = [
                    shift_span(span, offset=-partition.start) for span in arg_spans
                ]
                # map character spans to token spans
                try:
                    arg_token_spans = [
                        get_aligned_token_span(
                            encoding=encoding,
                            char_span=arg,
                        )
                        for arg in arg_spans_partition
                    ]
                # Check if the mapping was successful. It may fail (and is None) if any argument start or end does not
                # match a token start or end, respectively.
                except SpanNotAlignedWithTokenException as e:
                    span_original = shift_span(e.span, offset=partition.start)
                    # the span is not attached because we shifted it above, so we can not use str(e.span)
                    span_text = document.text[span_original.start : span_original.end]
                    logger.warning(
                        f"doc.id={document.id}: Skipping invalid example, cannot get argument token slice for "
                        f'{span_original}: "{span_text}"'
                    )
                    self.collect_relation("skipped_args_not_aligned", rel)
                    continue

                # create the argument objects
                args = [
                    RelationArgument(
                        entity=span,
                        role=role,
                        token_span=token_span,
                        add_type_to_marker=self.add_type_to_marker,
                        marker_factory=self.marker_factory,
                    )
                    for span, role, token_span in zip(arg_spans, arg_roles, arg_token_spans)
                ]

                if self.max_argument_distance_tokens is not None:
                    token_distances = []
                    for idx1 in range(len(args) - 1):
                        for idx in range(idx1 + 1, len(args)):
                            arg1 = args[idx1]
                            arg2 = args[idx]
                            dist = span_distance(
                                (arg1.token_span.start, arg1.token_span.end),
                                (arg2.token_span.start, arg2.token_span.end),
                                self.max_argument_distance_type_tokens,
                            )
                            token_distances.append(dist)
                    if len(token_distances) > 0:
                        if self.max_argument_distance_type_tokens == "outer":
                            max_dist = max(token_distances)
                        elif self.max_argument_distance_type_tokens == "inner":
                            if len(args) > 2:
                                raise NotImplementedError(
                                    f"max_argument_distance_type_tokens={self.max_argument_distance_type_tokens} "
                                    f"is not supported for relations with more than 2 arguments"
                                )
                            max_dist = max(token_distances)
                        else:
                            raise NotImplementedError(
                                f"max_argument_distance_type_tokens={self.max_argument_distance_type_tokens} "
                                f"is not supported"
                            )
                        if max_dist > self.max_argument_distance_tokens:
                            self.collect_relation("skipped_argument_distance_tokens", rel)
                            continue

                input_ids = encoding["input_ids"]

                entity_tags = None
                if self.add_entity_tags_to_input:
                    entity_spans_partition = [
                        shift_span(span, offset=-partition.start) for span in entities
                    ]
                    entity_token_spans = []
                    for span in entity_spans_partition:
                        try:
                            entity_token_spans.append(
                                get_aligned_token_span(
                                    encoding=encoding,
                                    char_span=span,
                                )
                            )
                        except SpanNotAlignedWithTokenException as e:
                            span_original = shift_span(e.span, offset=partition.start)
                            span_text = document.text[span_original.start : span_original.end]
                            logger.warning(
                                f"doc.id={document.id}: Skipping invalid example, cannot get entity token slice for "
                                f'{span_original}: "{span_text}"'
                            )
                            self.collect_relation("skipped_entity_not_aligned", rel)
                            continue

                    entity_tags = bio_encode_spans(
                        spans=[
                            (span.start, span.end, getattr(span, "label", "ENTITY"))
                            for span in entity_token_spans
                        ],
                        total_length=len(input_ids),
                        label2idx={
                            label: idx for idx, label in enumerate(self.entity_labels or [])
                        },
                    )

                # windowing: we restrict the input to a window of a maximal size (max_window) with the arguments
                # of the candidate relation in the center (as much as possible)
                if self.max_window is not None:
                    # The actual number of tokens needs to be lower than max_window because we add two
                    # marker tokens (before / after) each argument and the default special tokens
                    # (e.g. CLS and SEP).
                    max_tokens = self.max_window - self.tokenizer.num_special_tokens_to_add()
                    if self.insert_markers:
                        max_tokens -= len(args) * 2
                    # if we add the markers also to the end, this decreases the available window again by
                    # two tokens (marker + sep) per argument
                    if self.append_markers:
                        # TODO: add test case for this
                        max_tokens -= len(args) * 2

                    if self.allow_discontinuous_text:
                        if entity_tags is not None:
                            raise NotImplementedError(
                                "allow_discontinuous_text=True is not yet supported with add_entity_tags_to_input=True"
                            )

                        max_tokens_per_argument = max_tokens // len(args)
                        max_tokens_per_argument -= len(self.glue_token_ids)
                        if any(
                            arg.token_span.end - arg.token_span.start > max_tokens_per_argument
                            for arg in args
                        ):
                            self.collect_relation("skipped_too_long_argument", rel)
                            continue

                        mask = np.zeros_like(input_ids)
                        for arg in args:
                            # if the input is already fully covered by one argument frame, we keep everything
                            if len(input_ids) <= max_tokens_per_argument:
                                mask[:] = 1
                                break
                            arg_center = (arg.token_span.end + arg.token_span.start) // 2
                            arg_frame_start = arg_center - max_tokens_per_argument // 2
                            # shift the frame to the right if it is out of bounds
                            if arg_frame_start < 0:
                                arg_frame_start = 0
                            arg_frame_end = arg_frame_start + max_tokens_per_argument
                            # shift the frame to the left if it is out of bounds
                            # Note that this can not cause to have arg_frame_start < 0 because we already
                            # checked that the frame is not larger than the input.
                            if arg_frame_end > len(input_ids):
                                arg_frame_end = len(input_ids)
                                arg_frame_start = arg_frame_end - max_tokens_per_argument
                            # still, a sanity check
                            if arg_frame_start < 0:
                                raise ValueError(
                                    f"arg_frame_start={arg_frame_start} < 0 after adjusting arg_frame_end={arg_frame_end}"
                                )
                            mask[arg_frame_start:arg_frame_end] = 1
                        offsets = np.cumsum(mask != 1)
                        arg_cluster_offset_values = set()
                        # sort by start indices
                        args_sorted = sorted(args, key=lambda x: x.token_span.start)
                        for arg in args_sorted:
                            offset = offsets[arg.token_span.start]
                            arg_cluster_offset_values.add(offset)
                            arg.shift_token_span(-offset)
                            # shift back according to inserted glue patterns
                            num_glues = len(arg_cluster_offset_values) - 1
                            arg.shift_token_span(num_glues * len(self.glue_token_ids))

                        new_input_ids: List[int] = []
                        for arg_cluster_offset_value in sorted(arg_cluster_offset_values):
                            if len(new_input_ids) > 0:
                                new_input_ids.extend(self.glue_token_ids)
                            segment_mask = offsets == arg_cluster_offset_value
                            segment_input_ids = [
                                input_id
                                for input_id, keep in zip(input_ids, mask & segment_mask)
                                if keep
                            ]
                            new_input_ids.extend(segment_input_ids)

                        input_ids = new_input_ids
                    else:
                        # the slice from the beginning of the first entity to the end of the second is required
                        slice_required = (
                            min(arg.token_span.start for arg in args),
                            max(arg.token_span.end for arg in args),
                        )
                        window_slice = get_window_around_slice(
                            slice=slice_required,
                            max_window_size=max_tokens,
                            available_input_length=len(input_ids),
                        )
                        # this happens if slice_required (all arguments) does not fit into max_tokens (the available window)
                        if window_slice is None:
                            self.collect_relation("skipped_too_long", rel)
                            continue

                        window_start, window_end = window_slice
                        input_ids = input_ids[window_start:window_end]

                        if entity_tags is not None:
                            entity_tags = entity_tags[window_start:window_end]

                        for arg in args:
                            arg.shift_token_span(-window_start)

                # collect all markers with their target positions, the source argument, and
                marker_ids_with_positions = []
                for arg in args:
                    marker_ids_with_positions.append(
                        (
                            self.argument_markers_to_id[arg.as_start_marker],
                            arg.token_span.start,
                            arg,
                            START,
                        )
                    )
                    marker_ids_with_positions.append(
                        (
                            self.argument_markers_to_id[arg.as_end_marker],
                            arg.token_span.end,
                            arg,
                            END,
                        )
                    )

                # create new input ids with the markers inserted and collect new mention offsets
                input_ids_with_markers = list(input_ids)
                offset = 0
                arg_start_indices = [-1] * len(self.argument_role2idx)
                arg_end_indices = [-1] * len(self.argument_role2idx)
                marker_ids_with_positions_sorted = sorted(
                    marker_ids_with_positions, key=lambda id_pos: id_pos[1]
                )
                for (
                    marker_id,
                    token_position,
                    arg,
                    marker_type,
                ) in marker_ids_with_positions_sorted:
                    if self.insert_markers:
                        input_ids_with_markers = (
                            input_ids_with_markers[: token_position + offset]
                            + [marker_id]
                            + input_ids_with_markers[token_position + offset :]
                        )
                        if entity_tags is not None:
                            entity_tags = (
                                entity_tags[: token_position + offset]
                                + [0]
                                + entity_tags[token_position + offset :]
                            )
                        offset += 1
                    if self.add_argument_indices_to_input or self.add_argument_tags_to_input:
                        idx = self.argument_role2idx[arg.role]
                        if marker_type == START:
                            if arg_start_indices[idx] != -1:
                                # TODO: add test case for this
                                raise ValueError(
                                    f"Trying to overwrite arg_start_indices[{idx}]={arg_start_indices[idx]} with "
                                    f"{token_position + offset} for document {document.id}"
                                )
                            arg_start_indices[idx] = token_position + offset
                        elif marker_type == END:
                            if arg_end_indices[idx] != -1:
                                # TODO: add test case for this
                                raise ValueError(
                                    f"Trying to overwrite arg_start_indices[{idx}]={arg_end_indices[idx]} with "
                                    f"{token_position + offset} for document {document.id}"
                                )
                            # -1 to undo the additional offset for the end marker which does not
                            # affect the mention offset
                            arg_end_indices[idx] = (
                                token_position + offset - (1 if self.insert_markers else 0)
                            )

                if self.append_markers:
                    if self.tokenizer.sep_token is None:
                        # TODO: add test case for this
                        raise ValueError("append_markers is True, but tokenizer has no sep_token")
                    sep_token_id = self.tokenizer.vocab[self.tokenizer.sep_token]
                    for arg in args:
                        if without_special_tokens:
                            # TODO: add test case for this
                            input_ids_with_markers.append(sep_token_id)
                            input_ids_with_markers.append(
                                self.argument_markers_to_id[arg.as_append_marker]
                            )
                        else:
                            input_ids_with_markers.append(
                                self.argument_markers_to_id[arg.as_append_marker]
                            )
                            input_ids_with_markers.append(sep_token_id)
                        if entity_tags is not None:
                            entity_tags.append(0)
                            entity_tags.append(0)

                # when windowing is used, we have to add the special tokens manually
                if without_special_tokens:
                    original_input_ids_with_markers = input_ids_with_markers
                    input_ids_with_markers = self.tokenizer.build_inputs_with_special_tokens(
                        token_ids_0=input_ids_with_markers
                    )
                    if self.add_argument_indices_to_input or self.add_argument_tags_to_input:
                        # get the number of prefix tokens
                        index_offset = find_sublist(
                            sub=original_input_ids_with_markers, bigger=input_ids_with_markers
                        )
                        if index_offset == -1:
                            raise ValueError(
                                f"Could not find the original tokens in the prefixed tokens for document {document.id}"
                            )
                        arg_start_indices = [
                            idx + index_offset if idx != -1 else -1 for idx in arg_start_indices
                        ]
                        arg_end_indices = [
                            idx + index_offset if idx != -1 else -1 for idx in arg_end_indices
                        ]
                    if entity_tags is not None:
                        special_tokens_mask = self.tokenizer.get_special_tokens_mask(
                            token_ids_0=input_ids_with_markers, already_has_special_tokens=True
                        )
                        entity_tags_with_special = self.tokenizer.build_inputs_with_special_tokens(
                            token_ids_0=entity_tags
                        )
                        entity_tags = [
                            tag if not is_special else 0
                            for tag, is_special in zip(
                                entity_tags_with_special, special_tokens_mask
                            )
                        ]

                inputs = {"input_ids": input_ids_with_markers}
                if self.add_argument_indices_to_input:
                    inputs["pooler_start_indices"] = arg_start_indices
                    inputs["pooler_end_indices"] = arg_end_indices
                if self.add_argument_tags_to_input:
                    # create bio-encoded tags for the arguments
                    # using arg_start_indices, arg_end_indices, and marker_ids_with_positions_sorted
                    argument_spans = [
                        (
                            arg_start_indices[self.argument_role2idx[arg.role]],
                            arg_end_indices[self.argument_role2idx[arg.role]],
                            arg.role,
                        )
                        for marker_id, token_position, arg, marker_type in marker_ids_with_positions_sorted
                    ]
                    argument_tag_ids = bio_encode_spans(
                        spans=argument_spans,
                        total_length=len(input_ids_with_markers),
                        label2idx=self.argument_role2idx,
                    )
                    inputs["argument_tags"] = argument_tag_ids

                if entity_tags is not None:
                    inputs["entity_tags"] = entity_tags

                task_encodings.append(
                    TaskEncoding(
                        document=document,
                        inputs=inputs,
                        metadata=({"candidate_annotation": rel}),
                    )
                )

                self.collect_relation("used", rel)

        return task_encodings

    def _maybe_log_example(
        self,
        task_encoding: TaskEncodingType,
        target: TargetEncodingType,
    ):
        """Maybe log the example."""

        # log the first n examples
        if self._logged_examples_counter < self.log_first_n_examples:
            input_ids = task_encoding.inputs["input_ids"]
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            target_labels = [self.id_to_label[label_id] for label_id in target]
            logger.info("*** Example ***")
            logger.info("doc id: %s", task_encoding.document.id)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("Expected label: %s (ids = %s)", target_labels, target)

            self._logged_examples_counter += 1

    def encode_target(
        self,
        task_encoding: TaskEncodingType,
    ) -> TargetEncodingType:
        candidate_annotation = task_encoding.metadata["candidate_annotation"]
        if isinstance(candidate_annotation, (BinaryRelation, NaryRelation)):
            labels = [candidate_annotation.label]
        else:
            raise NotImplementedError(
                f"encoding the target with a candidate_annotation of another type than BinaryRelation or"
                f"NaryRelation is not yet supported. candidate_annotation has the type: "
                f"{type(candidate_annotation)}"
            )
        target = [self.label_to_id[label] for label in labels]

        self._maybe_log_example(task_encoding=task_encoding, target=target)

        return target

    def unbatch_output(self, model_output: ModelTargetType) -> Sequence[TaskOutputType]:
        unbatched_output = []
        if self.multi_label:
            raise NotImplementedError
        else:
            label_ids = model_output["labels"].detach().cpu().tolist()
            probabilities = model_output["probabilities"].detach().cpu().tolist()
            for batch_idx in range(len(label_ids)):
                label_id = label_ids[batch_idx]
                result: TaskOutputType = {
                    "labels": [self.id_to_label[label_id]],
                    "probabilities": [probabilities[batch_idx][label_id]],
                }
                unbatched_output.append(result)

        return unbatched_output

    def create_annotations_from_output(
        self,
        task_encoding: TaskEncodingType,
        task_output: TaskOutputType,
    ) -> Iterator[Tuple[str, Union[BinaryRelation, MultiLabeledBinaryRelation, NaryRelation]]]:
        candidate_annotation = task_encoding.metadata["candidate_annotation"]
        new_annotation: Union[BinaryRelation, MultiLabeledBinaryRelation, NaryRelation]
        if self.multi_label:
            raise NotImplementedError
        else:
            label = task_output["labels"][0]
            probability = (
                task_output["probabilities"][0] if "probabilities" in task_output else 1.0
            )
            if isinstance(candidate_annotation, BinaryRelation):
                head = candidate_annotation.head
                tail = candidate_annotation.tail
                # Reverse predicted reversed relations back. Serialization will remove any duplicated relations.
                if self.add_reversed_relations:
                    # TODO: add test case for this
                    if label.endswith(self.reversed_relation_label_suffix):
                        label = label[: -len(self.reversed_relation_label_suffix)]
                        head, tail = tail, head
                    # If the predicted label is symmetric, we sort the arguments by its center.
                    elif label in self.symmetric_relations and self.reverse_symmetric_relations:
                        if not (isinstance(head, Span) and isinstance(tail, Span)):
                            raise ValueError(
                                f"the taskmodule expects the relation arguments of the candidate_annotation"
                                f"to be of type Span, but got head of type: {type(head)} and tail of type: "
                                f"{type(tail)}"
                            )
                        # use a unique order for the arguments: sort by start and end positions
                        head, tail = sorted([head, tail], key=lambda span: (span.start, span.end))
                new_annotation = BinaryRelation(
                    head=head, tail=tail, label=label, score=probability
                )
            elif isinstance(candidate_annotation, NaryRelation):
                # TODO: add test case for this
                if self.add_reversed_relations:
                    raise ValueError("can not reverse a NaryRelation")
                new_annotation = NaryRelation(
                    arguments=candidate_annotation.arguments,
                    roles=candidate_annotation.roles,
                    label=label,
                    score=probability,
                )
            else:
                raise NotImplementedError(
                    f"creating a new annotation from a candidate_annotation of another type than BinaryRelation is "
                    f"not yet supported. candidate_annotation has the type: {type(candidate_annotation)}"
                )

            new_annotation_args = get_relation_argument_spans_and_roles(new_annotation)
            arg_roles, arg_spans = zip(*new_annotation_args)
            arg_labels = tuple(getattr(ann, "label") for ann in arg_spans)

            # Create annotation only if 1. and 2. are fulfilled:
            if (
                # 1. the label is not the no-relation-label,
                label != self.none_label
                #    or we did not create candidate relations,
                or not self.add_candidate_relations
            ) and (
                # 2. the argument_and_relation_type_whitelist is not set,
                self.argument_and_relation_type_whitelist is None
                #    or the label and argument types are in the whitelist
                or arg_labels in self.argument_and_relation_type_whitelist.get(label, {})
            ):
                yield self.relation_annotation, new_annotation

    def _get_global_attention(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        # we want to have global attention on all marker tokens and the cls token
        positive_token_ids = list(self.argument_markers_to_id.values()) + [
            self.tokenizer.cls_token_id
        ]
        global_attention_mask = construct_mask(
            input_ids=input_ids, positive_ids=positive_token_ids
        )
        return global_attention_mask

    def collate(
        self, task_encodings: Sequence[TaskEncodingType]
    ) -> Tuple[ModelInputType, Optional[ModelTargetType]]:
        input_features = [
            {"input_ids": task_encoding.inputs["input_ids"]} for task_encoding in task_encodings
        ]

        inputs: Dict[str, torch.LongTensor] = self.tokenizer.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        if self.add_argument_tags_to_input:
            argument_tags = [
                {"input_ids": task_encoding.inputs["argument_tags"]}
                for task_encoding in task_encodings
            ]
            argument_tags_padded = self.tokenizer.pad(
                argument_tags,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
            # increase all values by 1 because 0 is used for padding
            inputs["argument_tags"] = argument_tags_padded["input_ids"] + 1
            # overwrite padding with 0
            inputs["argument_tags"][argument_tags_padded["attention_mask"] == 0] = 0

        if self.add_entity_tags_to_input:
            entity_tags = [
                {"input_ids": task_encoding.inputs["entity_tags"]}
                for task_encoding in task_encodings
            ]
            entity_tags_padded = self.tokenizer.pad(
                entity_tags,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
            # increase all values by 1 because 0 is used for padding
            inputs["entity_tags"] = entity_tags_padded["input_ids"] + 1
            # overwrite padding with 0
            inputs["entity_tags"][entity_tags_padded["attention_mask"] == 0] = 0

        if self.add_argument_indices_to_input:
            pooler_start_indices = torch.tensor(
                [task_encoding.inputs["pooler_start_indices"] for task_encoding in task_encodings]
            )
            assert isinstance(pooler_start_indices, LongTensor)
            inputs["pooler_start_indices"] = pooler_start_indices
            pooler_end_indices = torch.tensor(
                [task_encoding.inputs["pooler_end_indices"] for task_encoding in task_encodings]
            )
            assert isinstance(pooler_end_indices, LongTensor)
            inputs["pooler_end_indices"] = pooler_end_indices

        if self.add_global_attention_mask_to_input:
            inputs["global_attention_mask"] = self._get_global_attention(
                input_ids=inputs["input_ids"]
            )

        if not task_encodings[0].has_targets:
            return inputs, None

        target_list: List[TargetEncodingType] = [
            task_encoding.targets for task_encoding in task_encodings
        ]
        targets = torch.tensor(target_list, dtype=torch.long)

        if not self.multi_label:
            targets = targets.flatten()

        assert isinstance(targets, LongTensor)
        return inputs, {"labels": targets}

    def configure_model_metric(self, stage: str) -> MetricCollection:
        if self.label_to_id is None:
            raise ValueError(
                "The taskmodule has not been prepared yet, so label_to_id is not known. "
                "Please call taskmodule.prepare(documents) before configuring the model metric "
                "or pass the labels to the taskmodule constructor an call taskmodule.post_prepare()."
            )
        # we use the length of label_to_id because that contains the none_label (in contrast to labels)
        labels = [self.id_to_label[i] for i in range(len(self.label_to_id))]
        common_metric_kwargs: dict[str, Any] = {
            "num_classes": len(labels),
            "task": "multilabel" if self.multi_label else "multiclass",
        }
        return MetricCollection(
            {
                "with_tn": WrappedMetricWithPrepareFunction(
                    metric=MetricCollection(
                        {
                            "micro/f1": F1Score(average="micro", **common_metric_kwargs),
                            "macro/f1": F1Score(average="macro", **common_metric_kwargs),
                            "f1_per_label": ClasswiseWrapper(
                                F1Score(average=None, **common_metric_kwargs),
                                labels=labels,
                                postfix="/f1",
                            ),
                        }
                    ),
                    prepare_function=_get_labels,
                ),
                # We can not easily calculate the macro f1 here, because
                # F1Score with average="macro" would still include the none_label.
                "micro/f1_without_tn": WrappedMetricWithPrepareFunction(
                    metric=F1Score(average="micro", **common_metric_kwargs),
                    prepare_together_function=partial(
                        _get_labels_together_remove_none_label,
                        none_idx=self.label_to_id[self.none_label],
                    ),
                ),
            }
        )
