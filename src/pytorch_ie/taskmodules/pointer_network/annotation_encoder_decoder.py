import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from pie_documents.annotations import BinaryRelation, LabeledSpan, Span

from pytorch_ie.taskmodules.common import AnnotationEncoderDecoder, DecodingException

logger = logging.getLogger(__name__)


class DecodingLengthException(DecodingException[List[int]]):
    identifier = "len"


class DecodingOrderException(DecodingException[List[int]]):
    identifier = "order"


class DecodingSpanOverlapException(DecodingException[List[int]]):
    identifier = "overlap"


class DecodingLabelException(DecodingException[List[int]]):
    identifier = "label"


class DecodingNegativeIndexException(DecodingException[List[int]]):
    identifier = "index"


KEY_INVALID_CORRECT = "correct"


class SpanEncoderDecoder(AnnotationEncoderDecoder[Span, List[int]]):
    def __init__(self, exclusive_end: bool = True):
        self.exclusive_end = exclusive_end

    def encode(self, annotation: Span, metadata: Optional[Dict[str, Any]] = None) -> List[int]:
        end_idx = annotation.end
        if not self.exclusive_end:
            end_idx -= 1
        return [annotation.start, end_idx]

    def decode(self, encoding: List[int], metadata: Optional[Dict[str, Any]] = None) -> Span:
        if len(encoding) != 2:
            raise DecodingLengthException(
                f"two values are required to decode as Span, but encoding has length {len(encoding)}",
                encoding=encoding,
            )
        end_idx = encoding[1]
        if not self.exclusive_end:
            end_idx += 1
        if end_idx < encoding[0]:
            raise DecodingOrderException(
                f"end index can not be smaller than start index, but got: start={encoding[0]}, "
                f"end={end_idx}",
                encoding=encoding,
            )
        if any(idx < 0 for idx in encoding):
            raise DecodingNegativeIndexException(
                f"indices must be positive, but got: {encoding}", encoding=encoding
            )
        return Span(start=encoding[0], end=end_idx)


class SpanEncoderDecoderWithOffset(SpanEncoderDecoder):
    def __init__(self, offset: int, **kwargs):
        super().__init__(**kwargs)
        self.offset = offset

    def encode(self, annotation: Span, metadata: Optional[Dict[str, Any]] = None) -> List[int]:
        encoding = super().encode(annotation=annotation, metadata=metadata)
        return [x + self.offset for x in encoding]

    def decode(self, encoding: List[int], metadata: Optional[Dict[str, Any]] = None) -> Span:
        encoding = [x - self.offset for x in encoding]
        return super().decode(encoding=encoding, metadata=metadata)


class LabeledSpanEncoderDecoder(AnnotationEncoderDecoder[LabeledSpan, List[int]]):
    def __init__(
        self,
        span_encoder_decoder: AnnotationEncoderDecoder[Span, List[int]],
        label2id: Dict[str, int],
        mode: str,
    ):
        self.span_encoder_decoder = span_encoder_decoder
        self.label2id = label2id
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        self.mode = mode

    def encode(
        self, annotation: LabeledSpan, metadata: Optional[Dict[str, Any]] = None
    ) -> List[int]:
        encoded_span = self.span_encoder_decoder.encode(annotation=annotation, metadata=metadata)
        encoded_label = self.label2id[annotation.label]
        if self.mode == "indices_label":
            return encoded_span + [encoded_label]
        elif self.mode == "label_indices":
            return [encoded_label] + encoded_span
        else:
            raise ValueError(f"unknown mode: {self.mode}")

    def decode(
        self, encoding: List[int], metadata: Optional[Dict[str, Any]] = None
    ) -> LabeledSpan:
        if self.mode == "label_indices":
            encoded_label = encoding[0]
            encoded_span = encoding[1:]
        elif self.mode == "indices_label":
            encoded_label = encoding[-1]
            encoded_span = encoding[:-1]
        else:
            raise ValueError(f"unknown mode: {self.mode}")

        decoded_span = self.span_encoder_decoder.decode(encoding=encoded_span, metadata=metadata)
        if encoded_label not in self.id2label:
            raise DecodingLabelException(
                f"unknown label id: {encoded_label} (label2id: {self.label2id})", encoding=encoding
            )
        result = LabeledSpan(
            start=decoded_span.start,
            end=decoded_span.end,
            label=self.id2label[encoded_label],
        )
        return result


class BinaryRelationEncoderDecoder(AnnotationEncoderDecoder[BinaryRelation, List[int]]):
    def __init__(
        self,
        head_encoder_decoder: AnnotationEncoderDecoder[Span, List[int]],
        tail_encoder_decoder: AnnotationEncoderDecoder[Span, List[int]],
        label2id: Dict[str, int],
        mode: str,
        loop_dummy_relation_name: Optional[str] = None,
        none_label: Optional[str] = None,
    ):
        self.head_encoder_decoder = head_encoder_decoder
        self.tail_encoder_decoder = tail_encoder_decoder
        self.loop_dummy_relation_name = loop_dummy_relation_name
        self.none_label = none_label
        self.label2id = label2id
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        self.mode = mode

    def encode(
        self, annotation: BinaryRelation, metadata: Optional[Dict[str, Any]] = None
    ) -> List[int]:
        encoded_head = self.head_encoder_decoder.encode(annotation=annotation.head)
        encoded_tail = self.tail_encoder_decoder.encode(annotation=annotation.tail)

        if (
            self.loop_dummy_relation_name is not None
            and annotation.label == self.loop_dummy_relation_name
        ):
            if annotation.head != annotation.tail:
                raise ValueError(
                    f"expected head == tail for loop_dummy_relation, but got: {annotation.head}, "
                    f"{annotation.tail}"
                )
            if self.none_label is None:
                raise ValueError(
                    f"loop_dummy_relation_name is set, but none_label is not set: {self.none_label}"
                )
            none_id = self.label2id[self.none_label]
            encoded_none_argument = [none_id, none_id, none_id]
            if self.mode == "head_tail_label":
                return encoded_head + encoded_none_argument + [none_id]
            elif self.mode == "tail_head_label":
                return encoded_tail + encoded_none_argument + [none_id]
            elif self.mode == "label_head_tail":
                return [none_id] + encoded_head + encoded_none_argument
            elif self.mode == "label_tail_head":
                return [none_id] + encoded_tail + encoded_none_argument
            else:
                raise ValueError(f"unknown mode: {self.mode}")
        else:
            encoded_label = self.label2id[annotation.label]
            if self.mode == "tail_head_label":
                return encoded_tail + encoded_head + [encoded_label]
            elif self.mode == "head_tail_label":
                return encoded_head + encoded_tail + [encoded_label]
            elif self.mode == "label_head_tail":
                return [encoded_label] + encoded_head + encoded_tail
            elif self.mode == "label_tail_head":
                return [encoded_label] + encoded_tail + encoded_head
            else:
                raise ValueError(f"unknown mode: {self.mode}")

    def is_single_span_label(self, label: str) -> bool:
        return self.none_label is not None and label == self.none_label

    def decode(
        self, encoding: List[int], metadata: Optional[Dict[str, Any]] = None
    ) -> BinaryRelation:
        if len(encoding) != 7:
            raise DecodingLengthException(
                f"seven values are required to decode as BinaryRelation, but the encoding has length {len(encoding)}",
                encoding=encoding,
            )
        if self.mode.endswith("_label"):
            encoded_label = encoding[6]
            encoded_arguments = encoding[:6]
            argument_mode = self.mode[: -len("_label")]
        elif self.mode.startswith("label_"):
            encoded_label = encoding[0]
            encoded_arguments = encoding[1:]
            argument_mode = self.mode[len("label_") :]
        else:
            raise ValueError(f"unknown mode: {self.mode}")
        if encoded_label not in self.id2label:
            raise DecodingLabelException(
                f"unknown label id: {encoded_label} (label2id: {self.label2id})", encoding=encoding
            )
        label = self.id2label[encoded_label]
        if self.is_single_span_label(label=label):
            if argument_mode == "head_tail":
                span_encoder = self.head_encoder_decoder
            elif argument_mode == "tail_head":
                span_encoder = self.tail_encoder_decoder
            else:
                raise ValueError(f"unknown argument mode: {argument_mode}")
            encoded_span = encoded_arguments[:3]
            span = span_encoder.decode(encoding=encoded_span, metadata=metadata)
            if self.loop_dummy_relation_name is None:
                raise ValueError(
                    f"loop_dummy_relation_name is not set, but none_label={self.none_label} "
                    f"was found in decoded encoding: {encoding} (label2id: {self.label2id}))"
                )
            rel = BinaryRelation(head=span, tail=span, label=self.loop_dummy_relation_name)
        else:
            if argument_mode == "head_tail":
                encoded_head = encoded_arguments[:3]
                encoded_tail = encoded_arguments[3:]
            elif argument_mode == "tail_head":
                encoded_tail = encoded_arguments[:3]
                encoded_head = encoded_arguments[3:]
            else:
                raise ValueError(f"unknown argument mode: {argument_mode}")
            head = self.head_encoder_decoder.decode(encoding=encoded_head, metadata=metadata)
            tail = self.tail_encoder_decoder.decode(encoding=encoded_tail, metadata=metadata)
            rel = BinaryRelation(head=head, tail=tail, label=label)

        return rel

    def build_decoding_constraints(
        self, partial_encoding: List[int]
    ) -> Tuple[Optional[Set[int]], Optional[Set[int]]]:
        """Given a partial encoding, build the constraints for the next encoding step.

        Returns:
            Tuple[Optional[Set[int]], Optional[Set[int]]]: A tuple of two sets of integers representing the allowed
                and disallowed next indices. The first set contains the allowed indices, and the second set contains
                the disallowed indices. If no constraints are needed, both sets can be None.
        """
        allowed = None
        disallowed = None

        if self.mode != "tail_head_label":
            raise NotImplementedError(
                f"build_decoder_constraints is not implemented for mode {self.mode}"
            )

        if self.none_label not in self.label2id:
            raise ValueError(
                f"none_label not found in label2id: {self.label2id} (none_label: {self.none_label})"
            )
        none_id = self.label2id[self.none_label]
        if self.head_encoder_decoder != self.tail_encoder_decoder:
            raise NotImplementedError(
                "head and tail encoder/decoder must be the same for build_decoder_constraints"
            )

        if not isinstance(self.head_encoder_decoder, LabeledSpanEncoderDecoder):
            raise NotImplementedError(
                "head and tail encoder/decoder must be LabeledSpanEncoderDecoder for build_decoder_constraints"
            )
        if not isinstance(
            self.head_encoder_decoder.span_encoder_decoder, SpanEncoderDecoderWithOffset
        ):
            raise NotImplementedError(
                "head and tail encoder/decoder must be SpanEncoderDecoderWithOffset for build_decoder_constraints"
            )
        pointer_offset = self.head_encoder_decoder.span_encoder_decoder.offset
        if self.head_encoder_decoder.mode != "indices_label":
            raise NotImplementedError(
                "head and tail encoder/decoder must be indices_label for build_decoder_constraints"
            )
        if (
            not isinstance(self.head_encoder_decoder.span_encoder_decoder, SpanEncoderDecoder)
            or self.head_encoder_decoder.span_encoder_decoder.exclusive_end
        ):
            raise NotImplementedError(
                "head and tail encoder/decoder must be exclusive_end for build_decoder_constraints"
            )
        span_ids = set(self.head_encoder_decoder.label2id.values())
        relation_ids = set(self.label2id.values()) - {self.label2id[self.none_label]}
        contains_none = none_id in partial_encoding
        idx = len(partial_encoding)
        if idx == 0:  # [] -> first span start or eos
            # Disallow all labels:
            disallowed = set(range(pointer_offset))
        elif idx == 1:  # [14] -> first span end
            # Allow all offsets greater than the span start.
            span_start = partial_encoding[-1]
            # result[span_start:] = 1
            disallowed = set(range(span_start))
            # Disallow the none label:
            disallowed.add(none_id)
        elif idx == 2:  # [14,14] -> first span label
            # Allow only span ids.
            allowed = span_ids
        elif idx == 3:  # [14,14,s1] -> second span start or none
            # Disallow overlap of first and second span:
            first_span_start = partial_encoding[0]
            first_span_end = partial_encoding[1] + 1
            disallowed = set(range(first_span_start, first_span_end))
            # Disallow all span labels:
            disallowed.update(span_ids)
            # Disallow all relation labels:
            disallowed.update(relation_ids)
            # But allow the none label:
            disallowed.discard(none_id)

        elif idx == 4:  # [14,14,s1,23] -> second span end or none
            # if we have a none label, allow only none
            if contains_none:
                allowed = {none_id}
            else:

                first_span_start = partial_encoding[0]
                # first_span_end = partial_encoding[1] + 1
                second_span_start = partial_encoding[-1]
                # if first span is after the second span,
                if second_span_start < first_span_start:
                    # just allow the offsets between the two spans:
                    allowed = set(range(second_span_start, first_span_start))
                else:
                    # otherwise, disallow all offsets before the second span start:
                    disallowed = set(range(second_span_start))

                    # Disallow all span labels:
                    disallowed.update(span_ids)
                    # Disallow all relation labels:
                    disallowed.update(relation_ids)

        elif idx == 5:  # [14,14,s1,23,25] -> second span label or none
            # if we have a none label, allow only none
            if contains_none:
                # result[none_id] = 1
                allowed = {none_id}
            else:
                # allow only span ids
                allowed = span_ids
        elif idx == 6:  # [14,14,s1,23,25,s2] -> relation label or none
            # if we have a none label, allow only none
            if contains_none:
                allowed = {none_id}
            else:
                # allow only relation ids
                allowed = relation_ids
        else:
            raise ValueError(
                f"unknown partial encoding length: {len(partial_encoding)} (encoding: {partial_encoding})"
            )

        return allowed, disallowed

    def parse(self, encoding: List[int]) -> Tuple[List[BinaryRelation], Dict[str, int], List[int]]:
        errors: Dict[str, int] = defaultdict(int)
        if self.none_label is None:
            raise ValueError(
                f"none_label is not set, but is required for parsing: {self.none_label}"
            )
        none_id = self.label2id[self.none_label]
        relation_ids = set(self.label2id.values()) - {none_id}
        encodings = []
        current_encoding: List[int] = []
        valid_encoding: BinaryRelation
        if len(encoding):
            for i in encoding:
                current_encoding.append(i)
                # An encoding is complete when it ends with a relation_id
                # or when it contains a none_id and has a length of 7
                if i in relation_ids or (i == none_id and len(current_encoding) == 7):
                    # try to decode the current relation encoding
                    try:
                        valid_encoding = self.decode(encoding=current_encoding)
                        encodings.append(valid_encoding)
                        errors[KEY_INVALID_CORRECT] += 1
                    except DecodingException as e:
                        errors[e.identifier] += 1

                    current_encoding = []

        return encodings, dict(errors), current_encoding
