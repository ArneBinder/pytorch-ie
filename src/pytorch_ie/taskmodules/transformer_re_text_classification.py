"""
workflow:
    Document
        -> (InputEncoding, TargetEncoding) -> TaskEncoding -> TaskBatchEncoding
            -> ModelBatchEncoding -> ModelBatchOutput
        -> TaskOutput
    -> Document
"""

import logging
from typing import (
    Any,
    Dict,
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
from transformers import AutoTokenizer
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy
from typing_extensions import TypeAlias

from pytorch_ie.annotations import (
    BinaryRelation,
    LabeledSpan,
    MultiLabeledBinaryRelation,
    NaryRelation,
    Span,
)
from pytorch_ie.core import AnnotationLayer, Document, TaskEncoding, TaskModule
from pytorch_ie.documents import (
    TextDocument,
    TextDocumentWithLabeledSpansAndBinaryRelations,
    TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
)
from pytorch_ie.models.transformer_text_classification import ModelOutputType, ModelStepInputType
from pytorch_ie.taskmodules.interface import ChangesTokenizerVocabSize
from pytorch_ie.utils.span import get_token_slice, is_contained_in
from pytorch_ie.utils.window import get_window_around_slice

InputEncodingType: TypeAlias = Dict[str, Any]
TargetEncodingType: TypeAlias = Sequence[int]

TaskEncodingType: TypeAlias = TaskEncoding[
    TextDocument,
    InputEncodingType,
    TargetEncodingType,
]


class TaskOutputType(TypedDict, total=False):
    labels: Sequence[str]
    probabilities: Sequence[float]


TaskModuleType: TypeAlias = TaskModule[
    TextDocument,
    InputEncodingType,
    TargetEncodingType,
    ModelStepInputType,
    ModelOutputType,
    TaskOutputType,
]


HEAD = "head"
TAIL = "tail"
START = "start"
END = "end"


logger = logging.getLogger(__name__)


class RelationArgument:
    def __init__(
        self,
        entity: LabeledSpan,
        role: str,
        token_span: Span,
        add_type_to_marker: bool,
        role_to_marker: Dict[str, str],
    ) -> None:
        self.entity = entity
        self.role_to_marker = role_to_marker
        if role not in self.role_to_marker:
            raise Exception(f"role={role} not in role_to_marker={role_to_marker}")
        self.role = role
        self.token_span = token_span
        self.add_type_to_marker = add_type_to_marker

    @property
    def as_start_marker(self) -> str:
        return self._get_marker(is_start=True)

    @property
    def as_end_marker(self) -> str:
        return self._get_marker(is_start=False)

    @property
    def role_marker(self) -> str:
        return self.role_to_marker[self.role]

    def _get_marker(self, is_start: bool = True) -> str:
        return f"[{'' if is_start else '/'}{self.role_marker}" + (
            f":{self.entity.label}]" if self.add_type_to_marker else "]"
        )

    @property
    def as_append_marker(self) -> str:
        return f"[{self.role_marker}={self.entity.label}]"

    def shift_token_span(self, value: int):
        self.token_span = Span(
            start=self.token_span.start + value, end=self.token_span.end + value
        )


@TaskModule.register()
class TransformerRETextClassificationTaskModule(TaskModuleType, ChangesTokenizerVocabSize):
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
    """

    PREPARED_ATTRIBUTES = ["label_to_id", "entity_labels"]

    def __init__(
        self,
        tokenizer_name_or_path: str,
        # this is deprecated, the target of the relation layer already specifies the entity layer
        entity_annotation: Optional[str] = None,
        relation_annotation: str = "binary_relations",
        create_relation_candidates: bool = False,
        partition_annotation: Optional[str] = None,
        none_label: str = "no_relation",
        padding: Union[bool, str, PaddingStrategy] = True,
        truncation: Union[bool, str, TruncationStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        multi_label: bool = False,
        label_to_id: Optional[Dict[str, int]] = None,
        add_type_to_marker: bool = False,
        argument_role_to_marker: Optional[Dict[str, str]] = None,
        single_argument_pair: bool = True,
        append_markers: bool = False,
        entity_labels: Optional[List[str]] = None,
        reversed_relation_label_suffix: Optional[str] = None,
        max_window: Optional[int] = None,
        log_first_n_examples: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if entity_annotation is not None:
            logger.warning(
                "The parameter entity_annotation is deprecated and will be discarded because it is not necessary "
                "anymore. The target of the relation layer already specifies the entity layer."
            )
        self.save_hyperparameters(ignore=["entity_annotation"])

        self.relation_annotation = relation_annotation
        self.create_relation_candidates = create_relation_candidates
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
        self.reversed_relation_label_suffix = reversed_relation_label_suffix
        self.max_window = max_window
        # overwrite None with 0 for backward compatibility
        self.log_first_n_examples = log_first_n_examples or 0

        if argument_role_to_marker is None:
            self.argument_role_to_marker = {HEAD: "H", TAIL: "T"}
        else:
            self.argument_role_to_marker = argument_role_to_marker

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

        self.argument_markers = None

        self._logged_examples_counter = 0

    @property
    def document_type(self) -> Optional[Type[TextDocument]]:
        dt: Type[TextDocument]
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
        if len(relations._targets) != 1:
            raise Exception(
                f"the relation layer is expected to target exactly one entity layer, but it has "
                f"the following targets: {relations._targets}"
            )
        entity_layer_name = relations._targets[0]
        return document[entity_layer_name]

    def _prepare(self, documents: Sequence[TextDocument]) -> None:
        entity_labels: Set[str] = set()
        relation_labels: Set[str] = set()
        for document in documents:
            relations: AnnotationLayer[BinaryRelation] = self.get_relation_layer(document)
            entities: AnnotationLayer[LabeledSpan] = self.get_entity_layer(document)

            for entity in entities:
                entity_labels.add(entity.label)

            for relation in relations:
                relation_labels.add(relation.label)

        if self.none_label in relation_labels:
            relation_labels.remove(self.none_label)

        self.label_to_id = {label: i + 1 for i, label in enumerate(sorted(relation_labels))}
        self.label_to_id[self.none_label] = 0

        self.entity_labels = sorted(entity_labels)

    def construct_argument_markers(self) -> List[str]:
        # ignore the typing because we know that this is only called on a prepared taskmodule,
        # i.e. self.entity_labels is already set by _prepare or __init__
        entity_labels: List[str] = self.entity_labels  # type: ignore
        argument_markers: Set[str] = set()
        for arg_role, role_marker in self.argument_role_to_marker.items():
            for arg_pos in [START, END]:
                is_start = arg_pos == START
                argument_markers.add(f"[{'' if is_start else '/'}{role_marker}]")
                if self.add_type_to_marker:
                    for entity_type in entity_labels:
                        argument_markers.add(
                            f"[{'' if is_start else '/'}{role_marker}"
                            f"{':' + entity_type if self.add_type_to_marker else ''}]"
                        )
                if self.append_markers:
                    for entity_type in entity_labels:
                        argument_markers.add(f"[{role_marker}={entity_type}]")

        return sorted(list(argument_markers))

    def _post_prepare(self):
        self.argument_markers = self.construct_argument_markers()
        self.tokenizer.add_tokens(self.argument_markers, special_tokens=True)

        self.argument_markers_to_id = {
            marker: self.tokenizer.vocab[marker] for marker in self.argument_markers
        }
        self.sep_token_id = self.tokenizer.vocab[self.tokenizer.sep_token]

        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

    def _create_relation_candidates(
        self,
        document: Document,
    ) -> List[BinaryRelation]:
        relation_candidates: List[BinaryRelation] = []
        relations: AnnotationLayer[BinaryRelation] = self.get_relation_layer(document)
        entities: AnnotationLayer[LabeledSpan] = self.get_entity_layer(document)
        arguments_to_relation = {(rel.head, rel.tail): rel for rel in relations}
        # iterate over all possible argument candidates
        for head in entities:
            for tail in entities:
                if head != tail:
                    # If there is no relation with the candidate arguments, we create a relation candidate with the
                    # none label. Otherwise, we use the existing relation.
                    candidate = arguments_to_relation.get(
                        (head, tail),
                        BinaryRelation(
                            head=head,
                            tail=tail,
                            label=self.none_label,
                            score=1.0,
                        ),
                    )
                    relation_candidates.append(candidate)

        return relation_candidates

    def encode_input(
        self,
        document: TextDocument,
    ) -> Optional[Union[TaskEncodingType, Sequence[TaskEncodingType],]]:
        relations: Sequence[BinaryRelation]
        if self.create_relation_candidates:
            relations = self._create_relation_candidates(document)
        else:
            relations = self.get_relation_layer(document)

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

            for rel in relations:
                arg_spans: List[LabeledSpan]
                if isinstance(rel, BinaryRelation):
                    if not isinstance(rel.head, LabeledSpan) or not isinstance(
                        rel.tail, LabeledSpan
                    ):
                        raise ValueError(
                            f"the taskmodule expects the relation arguments to be of type LabeledSpan, "
                            f"but got {type(rel.head)} and {type(rel.tail)}"
                        )
                    arg_spans = [rel.head, rel.tail]
                    arg_roles = [HEAD, TAIL]
                elif isinstance(rel, NaryRelation):
                    if any(not isinstance(arg, LabeledSpan) for arg in rel.arguments):
                        raise ValueError(
                            f"the taskmodule expects the relation arguments to be of type LabeledSpan, "
                            f"but got {[type(arg) for arg in rel.arguments]}"
                        )
                    arg_spans = list(rel.arguments)
                    arg_roles = list(rel.roles)
                else:
                    raise NotImplementedError(
                        f"the taskmodule does not yet support relations of type: {type(rel)}"
                    )

                # check if the argument spans are in the current partition
                if any(
                    not is_contained_in((arg.start, arg.end), (partition.start, partition.end))
                    for arg in arg_spans
                ):
                    continue

                # map character spans to token spans
                arg_token_slices_including_none = [
                    get_token_slice(
                        character_slice=(arg.start, arg.end),
                        char_to_token_mapper=encoding.char_to_token,
                        character_offset=partition.start,
                    )
                    for arg in arg_spans
                ]
                # Check if the mapping was successful. It may fail (and is None) if any argument start or end does not
                # match a token start or end, respectively.
                if any(token_slice is None for token_slice in arg_token_slices_including_none):
                    logger.warning(
                        f"Skipping invalid example {document.id}, cannot get argument token slice(s)"
                    )
                    continue

                # ignore the typing, because we checked for None above
                arg_token_slices: List[Tuple[int, int]] = arg_token_slices_including_none  # type: ignore

                # create the argument objects
                args = [
                    RelationArgument(
                        entity=span,
                        role=role,
                        token_span=Span(start=token_slice[0], end=token_slice[1]),
                        add_type_to_marker=self.add_type_to_marker,
                        role_to_marker=self.argument_role_to_marker,
                    )
                    for span, role, token_slice in zip(arg_spans, arg_roles, arg_token_slices)
                ]

                input_ids = encoding["input_ids"]

                # windowing: we restrict the input to a window of a maximal size (max_window) with the arguments
                # of the candidate relation in the center (as much as possible)
                if self.max_window is not None:
                    # The actual number of tokens needs to be lower than max_window because we add two
                    # marker tokens (before / after) each argument and the default special tokens
                    # (e.g. CLS and SEP).
                    max_tokens = (
                        self.max_window
                        - len(args) * 2
                        - self.tokenizer.num_special_tokens_to_add()
                    )
                    # if we add the markers also to the end, this decreases the available window again by
                    # two tokens (marker + sep) per argument
                    if self.append_markers:
                        max_tokens -= len(args) * 2
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
                        continue

                    window_start, window_end = window_slice
                    input_ids = input_ids[window_start:window_end]

                    for arg in args:
                        arg.shift_token_span(-window_start)

                # collect all markers with their target positions
                marker_ids_with_positions = []
                for arg in args:
                    marker_ids_with_positions.append(
                        (self.argument_markers_to_id[arg.as_start_marker], arg.token_span.start)
                    )
                    marker_ids_with_positions.append(
                        (self.argument_markers_to_id[arg.as_end_marker], arg.token_span.end)
                    )

                # create new input ids with the markers inserted
                input_ids_with_markers = list(input_ids)
                offset = 0
                for marker_id, token_position in sorted(
                    marker_ids_with_positions, key=lambda id_pos: id_pos[1]
                ):
                    input_ids_with_markers = (
                        input_ids_with_markers[: token_position + offset]
                        + [marker_id]
                        + input_ids_with_markers[token_position + offset :]
                    )
                    offset += 1

                if self.append_markers:
                    for arg in args:
                        if without_special_tokens:
                            input_ids_with_markers.append(self.sep_token_id)
                            input_ids_with_markers.append(
                                self.argument_markers_to_id[arg.as_append_marker]
                            )
                        else:
                            input_ids_with_markers.append(
                                self.argument_markers_to_id[arg.as_append_marker]
                            )
                            input_ids_with_markers.append(self.sep_token_id)

                # when windowing is used, we have to add the special tokens manually
                if without_special_tokens:
                    input_ids_with_markers = self.tokenizer.build_inputs_with_special_tokens(
                        token_ids_0=input_ids_with_markers
                    )

                task_encodings.append(
                    TaskEncoding(
                        document=document,
                        inputs={"input_ids": input_ids_with_markers},
                        metadata={"candidate_annotation": rel},
                    )
                )

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
        if isinstance(candidate_annotation, BinaryRelation):
            labels = [candidate_annotation.label]
        else:
            raise NotImplementedError(
                f"encoding the target with a candidate_annotation of another type than BinaryRelation is "
                f"not yet supported. candidate_annotation has the type: {type(candidate_annotation)}"
            )
        target = [self.label_to_id[label] for label in labels]

        self._maybe_log_example(task_encoding=task_encoding, target=target)

        return target

    def unbatch_output(self, model_output: ModelOutputType) -> Sequence[TaskOutputType]:
        logits = model_output["logits"]

        output_label_probs = logits.sigmoid() if self.multi_label else logits.softmax(dim=-1)
        output_label_probs = output_label_probs.detach().cpu().numpy()

        unbatched_output = []
        if self.multi_label:
            raise NotImplementedError
        else:
            label_ids = np.argmax(output_label_probs, axis=-1)
            for batch_idx, label_id in enumerate(label_ids):
                label = self.id_to_label[label_id]
                prob = float(output_label_probs[batch_idx, label_id])
                result: TaskOutputType = {
                    "labels": [label],
                    "probabilities": [prob],
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
            probability = task_output["probabilities"][0]
            if isinstance(candidate_annotation, BinaryRelation):
                head = candidate_annotation.head
                tail = candidate_annotation.tail
                # reverse any predicted reversed relations back
                if self.reversed_relation_label_suffix is not None and label.endswith(
                    self.reversed_relation_label_suffix
                ):
                    label = label[: -len(self.reversed_relation_label_suffix)]
                    head, tail = tail, head
                new_annotation = BinaryRelation(
                    head=head, tail=tail, label=label, score=probability
                )
            elif isinstance(candidate_annotation, NaryRelation):
                if self.reversed_relation_label_suffix is not None:
                    raise ValueError(f"can not reverse a NaryRelation")
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
            if label != self.none_label:
                yield self.relation_annotation, new_annotation

    def collate(self, task_encodings: Sequence[TaskEncodingType]) -> ModelStepInputType:
        input_features = [task_encoding.inputs for task_encoding in task_encodings]

        inputs: Dict[str, torch.Tensor] = self.tokenizer.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if not task_encodings[0].has_targets:
            return inputs, None

        target_list: List[TargetEncodingType] = [
            task_encoding.targets for task_encoding in task_encodings
        ]
        targets = torch.tensor(target_list, dtype=torch.int64)

        if not self.multi_label:
            targets = targets.flatten()

        return inputs, targets
