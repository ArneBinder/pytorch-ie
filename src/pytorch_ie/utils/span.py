import functools
import logging
from typing import (
    Callable,
    Counter,
    DefaultDict,
    Dict,
    List,
    MutableSequence,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from transformers import PreTrainedTokenizer

from pytorch_ie.annotations import LabeledSpan, Span

TypedSpan = Tuple[int, Tuple[int, int]]
TypedStringSpan = Tuple[str, Tuple[int, int]]

logger = logging.getLogger(__name__)


class InvalidTagSequence(Exception):
    def __init__(self, tag_sequence=None):
        super().__init__()
        self.tag_sequence = tag_sequence

    def __str__(self):
        return " ".join(self.tag_sequence)


def bio_tags_to_spans(
    tag_sequence: Sequence[str],
    classes_to_ignore: List[str] = None,
    include_ill_formed: bool = True,
) -> List[TypedStringSpan]:
    """
    Given a sequence corresponding to BIO tags, extracts spans.
    Spans are inclusive and can be of zero length, representing a single word span.
    Ill-formed spans are also included (i.e those which do not start with a "B-LABEL"),
    as otherwise it is possible to get a perfect precision score whilst still predicting
    ill-formed spans in addition to the correct spans. This function works properly when
    the spans are unlabeled (i.e., your labels are simply "B", "I", and "O").
    # Parameters
    tag_sequence : `List[str]`, required.
        The integer class labels for a sequence.
    classes_to_ignore : `List[str]`, optional (default = `None`).
        A list of string class labels `excluding` the bio tag
        which should be ignored when extracting spans.
    include_ill_formed: `bool`, optional (default = `True`).
        If this flag is enabled, include spans that do not start with "B". Otherwise, these are ignored.
    # Returns
    spans : `List[TypedStringSpan]`
        The typed, extracted spans from the sequence, in the format (label, (span_start, span_end)).
        Note that the label `does not` contain any BIO tag prefixes.
    """
    classes_to_ignore = classes_to_ignore or []
    spans: Set[Tuple[str, Tuple[int, int]]] = set()
    span_start = 0
    span_end = 0
    active_conll_tag = None
    for index, string_tag in enumerate(tag_sequence):
        # Actual BIO tag.
        bio_tag = string_tag[0]
        if bio_tag not in ["B", "I", "O"]:
            raise InvalidTagSequence(tag_sequence)
        conll_tag = string_tag[2:]
        if bio_tag == "O" or conll_tag in classes_to_ignore:
            # The span has ended.
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = None
            # We don't care about tags we are
            # told to ignore, so we do nothing.
            continue
        elif bio_tag == "B":
            # We are entering a new span; reset indices
            # and active tag to new span.
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = conll_tag
            span_start = index
            span_end = index
        elif bio_tag == "I" and conll_tag == active_conll_tag:
            # We're inside a span.
            span_end += 1
        else:
            # This is the case the bio label is an "I", but either:
            # 1) the span hasn't started - i.e. an ill formed span.
            # 2) The span is an I tag for a different conll annotation.
            # We'll process the previous span if it exists, but also
            # include this span. This is important, because otherwise,
            # a model may get a perfect F1 score whilst still including
            # false positive ill-formed spans.
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            if include_ill_formed:
                active_conll_tag = conll_tag
                span_start = index
                span_end = index
            else:
                active_conll_tag = None
                # We don't care about tags we are
                # told to ignore, so we do nothing.
                continue
    # Last token might have been a part of a valid span.
    if active_conll_tag is not None:
        spans.add((active_conll_tag, (span_start, span_end)))
    return list(spans)


def io_tags_to_spans(
    tag_sequence: List[str], classes_to_ignore: List[str] = None
) -> List[TypedStringSpan]:
    """
    Decode spans from simple IO encoding tag sequence, i.e. tags with an expected tag set of labels + "O".
    Create spans from maximal subsequences that have the same label.

    # Parameters
    tag_sequence : `List[str]`, required.
        The integer class labels for a sequence.
    classes_to_ignore : `List[str]`, optional (default = `None`).
        A list of string class labels `excluding` the bio tag
        which should be ignored when extracting spans.
    # Returns
    spans : `List[TypedStringSpan]`
        The typed, extracted spans from the sequence, in the format (label, (span_start, span_end)).
        Note that the label `does not` contain any BIO tag prefixes.
    """
    classes_to_ignore = classes_to_ignore or []
    spans: Set[Tuple[str, Tuple[int, int]]] = set()
    span_start = 0
    span_end = 0
    active_tag = None
    for index, string_tag in enumerate(tag_sequence):
        if string_tag == "O" or string_tag in classes_to_ignore:
            # The span has ended.
            if active_tag is not None:
                spans.add((active_tag, (span_start, span_end)))
            active_tag = None
            # We don't care about tags we are
            # told to ignore, so we do nothing.
            continue
        elif string_tag == active_tag:
            # We're inside a span.
            span_end += 1
        else:
            if active_tag is not None:
                spans.add((active_tag, (span_start, span_end)))
            active_tag = string_tag
            span_start = index
            span_end = index
    # Last token might have been a part of a valid span.
    if active_tag is not None:
        spans.add((active_tag, (span_start, span_end)))
    return list(spans)


def convert_span_annotations_to_tag_sequence(
    spans: Sequence[LabeledSpan],
    special_tokens_mask: Sequence[int],
    char_to_token_mapper: Callable[[int], Optional[int]],
    partition: Optional[Span] = None,
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
                statistics["skipped_unaligned"][span.label] += 1
            else:
                logger.warning(
                    f"Entity annotation does not start or end with a token, it will be skipped: {span}"
                )
            continue

        # negative numbers encode out-of-window tokens
        if start_idx < 0 or end_idx < 0:
            continue

        for j in range(start_idx, end_idx + 1):
            if tag_sequence[j] is not None and tag_sequence[j] != "O":
                # TODO: is ValueError a good exception type for this?
                raise ValueError(f"tag already assigned (current span has an overlap: {span})")
            prefix = "B" if j == start_idx else "I"
            tag_sequence[j] = f"{prefix}-{span.label}"

        if statistics is not None:
            statistics["added"][span.label] += 1

    return tag_sequence


def get_token_slice(
    character_slice: Tuple[int, int],
    char_to_token_mapper: Callable[[int], Optional[int]],
    character_offset: int = 0,
) -> Optional[Tuple[int, int]]:
    """
    Using an encoding to map a character slice to the respective token slice. If the slice start or end does
    not match a token start or end respectively, return None.
    """
    start = char_to_token_mapper(character_slice[0] - character_offset)
    before_end = char_to_token_mapper(character_slice[1] - 1 - character_offset)
    if start is None or before_end is None:
        return None
    return start, before_end + 1


def is_contained_in(start_end: Tuple[int, int], other_start_end: Tuple[int, int]) -> bool:
    return other_start_end[0] <= start_end[0] and start_end[1] <= other_start_end[1]


def has_overlap(start_end: Tuple[int, int], other_start_end: Tuple[int, int]):
    return (
        start_end[0] <= other_start_end[0] < start_end[1]
        or start_end[0] < other_start_end[1] <= start_end[1]
        or other_start_end[0] <= start_end[0] < other_start_end[1]
        or other_start_end[0] < start_end[1] <= other_start_end[1]
    )


def _char_to_token_mapper(
    char_idx: int,
    char_to_token_mapping: Dict[int, int],
    char_start: Optional[int] = None,
    char_end: Optional[int] = None,
) -> Optional[int]:
    if char_start is not None and char_idx < char_start:
        # return negative number to encode out-ot-window
        return -1
    if char_end is not None and char_idx >= char_end:
        # return negative number to encode out-ot-window
        return -2
    return char_to_token_mapping.get(char_idx, None)


def get_char_to_token_mapper(
    char_to_token_mapping: Dict[int, int],
    char_start: Optional[int] = None,
    char_end: Optional[int] = None,
) -> Callable[[int], Optional[int]]:
    return functools.partial(
        _char_to_token_mapper,
        char_to_token_mapping=char_to_token_mapping,
        char_start=char_start,
        char_end=char_end,
    )


def get_special_token_mask(token_ids_0: List[int], tokenizer: PreTrainedTokenizer) -> List[int]:
    # TODO: check why we can not just use tokenizer.get_special_tokens_mask()
    #  (this checks if token_ids_1 is not None and raises an exception)

    # exclude unknown token id since this indicate a real input token
    special_ids = set(tokenizer.all_special_ids) - {tokenizer.unk_token_id}
    return [1 if token_id in special_ids else 0 for token_id in token_ids_0]


def tokens_and_tags_to_text_and_labeled_spans(
    tokens: Sequence[str], tags: Sequence[str]
) -> Tuple[str, Sequence[LabeledSpan]]:
    start = 0
    token_offsets: List[Tuple[int, int]] = []
    for token in tokens:
        end = start + len(token)
        token_offsets.append((start, end))
        # we add a space after each token
        start = end + 1

    text = " ".join(tokens)

    spans: List[LabeledSpan] = []
    for label, (start, end) in bio_tags_to_spans(tags):
        spans.append(
            LabeledSpan(start=token_offsets[start][0], end=token_offsets[end][1], label=label)
        )

    return text, spans
