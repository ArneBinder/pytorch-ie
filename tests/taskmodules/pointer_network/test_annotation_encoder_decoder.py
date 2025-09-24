import pytest
from pie_documents.annotations import BinaryRelation, LabeledSpan, Span

from pytorch_ie.taskmodules.pointer_network.annotation_encoder_decoder import (
    BinaryRelationEncoderDecoder,
    DecodingLabelException,
    DecodingLengthException,
    DecodingNegativeIndexException,
    DecodingOrderException,
    LabeledSpanEncoderDecoder,
    SpanEncoderDecoder,
    SpanEncoderDecoderWithOffset,
)


@pytest.mark.parametrize("exclusive_end", [True, False])
def test_span_encoder_decoder(exclusive_end):
    """Test the SimpleSpanEncoderDecoder class."""

    encoder_decoder = SpanEncoderDecoder(exclusive_end)
    if exclusive_end:
        assert encoder_decoder.encode(Span(start=1, end=2)) == [1, 2]
        assert encoder_decoder.decode([1, 2]) == Span(start=1, end=2)
    else:
        assert encoder_decoder.encode(Span(start=1, end=2)) == [1, 1]
        assert encoder_decoder.decode([1, 1]) == Span(start=1, end=2)


def test_span_encoder_decoder_wrong_length():
    """Test the SimpleSpanEncoderDecoder class."""

    encoder_decoder = SpanEncoderDecoder()
    with pytest.raises(DecodingLengthException) as excinfo:
        encoder_decoder.decode([1])
    assert (
        str(excinfo.value)
        == "two values are required to decode as Span, but encoding has length 1"
    )
    assert excinfo.value.identifier == "len"

    with pytest.raises(DecodingLengthException) as excinfo:
        encoder_decoder.decode([1, 2, 3])
    assert (
        str(excinfo.value)
        == "two values are required to decode as Span, but encoding has length 3"
    )
    assert excinfo.value.identifier == "len"


def test_span_encoder_decoder_wrong_order():
    """Test the SimpleSpanEncoderDecoder class."""

    encoder_decoder = SpanEncoderDecoder()

    with pytest.raises(DecodingOrderException) as excinfo:
        encoder_decoder.decode([3, 2])
    assert (
        str(excinfo.value)
        == "end index can not be smaller than start index, but got: start=3, end=2"
    )
    assert excinfo.value.identifier == "order"

    # zero-length span
    span = encoder_decoder.decode([1, 1])
    assert span is not None


def test_span_encoder_decoder_wrong_offset():
    """Test the SimpleSpanEncoderDecoder class."""

    encoder_decoder = SpanEncoderDecoder()

    with pytest.raises(DecodingNegativeIndexException) as excinfo:
        encoder_decoder.decode([-1, 2])
    assert str(excinfo.value) == "indices must be positive, but got: [-1, 2]"
    assert excinfo.value.identifier == "index"


def test_span_encoder_decoder_with_offset():
    """Test the SpanEncoderDecoderWithOffset class."""

    encoder_decoder = SpanEncoderDecoderWithOffset(offset=1)

    assert encoder_decoder.encode(Span(start=1, end=2)) == [2, 3]
    assert encoder_decoder.decode([2, 3]) == Span(start=1, end=2)


@pytest.mark.parametrize("mode", ["indices_label", "label_indices"])
def test_labeled_span_encoder_decoder(mode):
    """Test the LabeledSpanEncoderDecoder class."""

    label2id = {"A": 0, "B": 1}
    encoder_decoder = LabeledSpanEncoderDecoder(
        span_encoder_decoder=SpanEncoderDecoderWithOffset(offset=len(label2id)),
        label2id=label2id,
        mode=mode,
    )

    if mode == "indices_label":
        assert encoder_decoder.encode(LabeledSpan(start=1, end=2, label="A")) == [3, 4, 0]
        assert encoder_decoder.decode([3, 4, 0]) == LabeledSpan(start=1, end=2, label="A")
    elif mode == "label_indices":
        assert encoder_decoder.encode(LabeledSpan(start=1, end=2, label="A")) == [0, 3, 4]
        assert encoder_decoder.decode([0, 3, 4]) == LabeledSpan(start=1, end=2, label="A")
    else:
        raise ValueError(f"unknown mode: {mode}")


@pytest.mark.parametrize("mode", ["indices_label", "label_indices"])
def test_labeled_span_encoder_decoder_wrong_label_encoding(mode):
    """Test the LabeledSpanEncoderDecoder class."""

    label2id = {"A": 0, "B": 1}
    encoder_decoder = LabeledSpanEncoderDecoder(
        span_encoder_decoder=SpanEncoderDecoderWithOffset(offset=len(label2id)),
        label2id=label2id,
        mode=mode,
    )

    if mode == "indices_label":
        with pytest.raises(DecodingLabelException) as excinfo:
            encoder_decoder.decode([2, 3, 4])
    elif mode == "label_indices":
        with pytest.raises(DecodingLabelException) as excinfo:
            encoder_decoder.decode([4, 2, 3])
    assert str(excinfo.value) == "unknown label id: 4 (label2id: {'A': 0, 'B': 1})"
    assert excinfo.value.identifier == "label"


def test_labeled_span_encoder_decoder_unknown_mode():
    """Test the LabeledSpanEncoderDecoder class."""

    label2id = {"A": 0, "B": 1}
    encoder_decoder = LabeledSpanEncoderDecoder(
        span_encoder_decoder=SpanEncoderDecoderWithOffset(offset=len(label2id)),
        label2id=label2id,
        mode="unknown",
    )
    with pytest.raises(ValueError) as excinfo:
        encoder_decoder.encode(LabeledSpan(start=1, end=2, label="A"))
    assert str(excinfo.value) == "unknown mode: unknown"

    with pytest.raises(ValueError) as excinfo:
        encoder_decoder.decode([0, 3, 4])
    assert str(excinfo.value) == "unknown mode: unknown"


@pytest.mark.parametrize(
    "mode", ["head_tail_label", "tail_head_label", "label_head_tail", "label_tail_head"]
)
def test_binary_relation_encoder_decoder(mode):
    """Test the BinaryRelationEncoderDecoder class."""

    label2id = {"A": 0, "B": 1, "C": 2}
    labeled_span_encoder_decoder = LabeledSpanEncoderDecoder(
        span_encoder_decoder=SpanEncoderDecoderWithOffset(offset=len(label2id)),
        label2id=label2id,
        mode="indices_label",
    )
    encoder_decoder = BinaryRelationEncoderDecoder(
        head_encoder_decoder=labeled_span_encoder_decoder,
        tail_encoder_decoder=labeled_span_encoder_decoder,
        label2id=label2id,
        mode=mode,
    )

    if mode == "head_tail_label":
        assert encoder_decoder.encode(
            BinaryRelation(
                head=LabeledSpan(start=1, end=2, label="A"),
                tail=LabeledSpan(start=3, end=4, label="B"),
                label="C",
            )
        ) == [4, 5, 0, 6, 7, 1, 2]
        assert encoder_decoder.decode([4, 5, 0, 6, 7, 1, 2]) == BinaryRelation(
            head=LabeledSpan(start=1, end=2, label="A"),
            tail=LabeledSpan(start=3, end=4, label="B"),
            label="C",
        )
    elif mode == "tail_head_label":
        assert encoder_decoder.encode(
            BinaryRelation(
                head=LabeledSpan(start=1, end=2, label="A"),
                tail=LabeledSpan(start=3, end=4, label="B"),
                label="C",
            )
        ) == [6, 7, 1, 4, 5, 0, 2]
        assert encoder_decoder.decode([6, 7, 1, 4, 5, 0, 2]) == BinaryRelation(
            head=LabeledSpan(start=1, end=2, label="A"),
            tail=LabeledSpan(start=3, end=4, label="B"),
            label="C",
        )
    elif mode == "label_head_tail":
        assert encoder_decoder.encode(
            BinaryRelation(
                head=LabeledSpan(start=1, end=2, label="A"),
                tail=LabeledSpan(start=3, end=4, label="B"),
                label="C",
            )
        ) == [2, 4, 5, 0, 6, 7, 1]
        assert encoder_decoder.decode([2, 4, 5, 0, 6, 7, 1]) == BinaryRelation(
            head=LabeledSpan(start=1, end=2, label="A"),
            tail=LabeledSpan(start=3, end=4, label="B"),
            label="C",
        )
    elif mode == "label_tail_head":
        assert encoder_decoder.encode(
            BinaryRelation(
                head=LabeledSpan(start=1, end=2, label="A"),
                tail=LabeledSpan(start=3, end=4, label="B"),
                label="C",
            )
        ) == [2, 6, 7, 1, 4, 5, 0]
        assert encoder_decoder.decode([2, 6, 7, 1, 4, 5, 0]) == BinaryRelation(
            head=LabeledSpan(start=1, end=2, label="A"),
            tail=LabeledSpan(start=3, end=4, label="B"),
            label="C",
        )


@pytest.mark.parametrize(
    "mode", ["head_tail_label", "tail_head_label", "label_head_tail", "label_tail_head"]
)
def test_binary_relation_encoder_decoder_loop_relation(mode):
    """Test the BinaryRelationEncoderDecoder class."""

    # we use different label2id for head and tail to test the case where the head and tail
    # have different label sets
    head_encoder_decoder = LabeledSpanEncoderDecoder(
        span_encoder_decoder=SpanEncoderDecoderWithOffset(offset=3),
        label2id={"A": 1, "B": 2},
        mode="indices_label",
    )
    tail_encoder_decoder = LabeledSpanEncoderDecoder(
        span_encoder_decoder=SpanEncoderDecoderWithOffset(offset=3),
        label2id={"A": -1, "B": -2},
        mode="indices_label",
    )
    encoder_decoder = BinaryRelationEncoderDecoder(
        head_encoder_decoder=head_encoder_decoder,
        tail_encoder_decoder=tail_encoder_decoder,
        label2id={"N": 3},
        mode=mode,
        loop_dummy_relation_name="L",
        none_label="N",
    )

    if mode == "head_tail_label":
        assert encoder_decoder.encode(
            BinaryRelation(
                head=LabeledSpan(start=1, end=2, label="A"),
                tail=LabeledSpan(start=1, end=2, label="A"),
                label="L",
            )
        ) == [4, 5, 1, 3, 3, 3, 3]
        assert encoder_decoder.decode([4, 5, 1, 3, 3, 3, 3]) == BinaryRelation(
            head=LabeledSpan(start=1, end=2, label="A"),
            tail=LabeledSpan(start=1, end=2, label="A"),
            label="L",
        )
    elif mode == "tail_head_label":
        assert encoder_decoder.encode(
            BinaryRelation(
                head=LabeledSpan(start=1, end=2, label="A"),
                tail=LabeledSpan(start=1, end=2, label="A"),
                label="L",
            )
        ) == [4, 5, -1, 3, 3, 3, 3]
        assert encoder_decoder.decode([4, 5, -1, 3, 3, 3, 3]) == BinaryRelation(
            head=LabeledSpan(start=1, end=2, label="A"),
            tail=LabeledSpan(start=1, end=2, label="A"),
            label="L",
        )
    elif mode == "label_head_tail":
        assert encoder_decoder.encode(
            BinaryRelation(
                head=LabeledSpan(start=1, end=2, label="A"),
                tail=LabeledSpan(start=1, end=2, label="A"),
                label="L",
            )
        ) == [3, 4, 5, 1, 3, 3, 3]
        assert encoder_decoder.decode([3, 4, 5, 1, 3, 3, 3]) == BinaryRelation(
            head=LabeledSpan(start=1, end=2, label="A"),
            tail=LabeledSpan(start=1, end=2, label="A"),
            label="L",
        )
    elif mode == "label_tail_head":
        assert encoder_decoder.encode(
            BinaryRelation(
                head=LabeledSpan(start=1, end=2, label="A"),
                tail=LabeledSpan(start=1, end=2, label="A"),
                label="L",
            )
        ) == [3, 4, 5, -1, 3, 3, 3]
        assert encoder_decoder.decode([3, 4, 5, -1, 3, 3, 3]) == BinaryRelation(
            head=LabeledSpan(start=1, end=2, label="A"),
            tail=LabeledSpan(start=1, end=2, label="A"),
            label="L",
        )
    else:
        raise ValueError(f"unknown mode: {mode}")


@pytest.mark.parametrize(
    "loop_dummy_relation_name,none_label",
    [("L", None), (None, "N")],
)
def test_binary_relation_encoder_decoder_only_loop_or_none_label_provided(
    loop_dummy_relation_name, none_label
):
    """Test the BinaryRelationEncoderDecoder class."""

    label2id = {"A": 0, "B": 1, "N": 2}
    labeled_span_encoder_decoder = LabeledSpanEncoderDecoder(
        span_encoder_decoder=SpanEncoderDecoderWithOffset(offset=len(label2id)),
        label2id=label2id,
        mode="indices_label",
    )
    encoder_decoder = BinaryRelationEncoderDecoder(
        head_encoder_decoder=labeled_span_encoder_decoder,
        tail_encoder_decoder=labeled_span_encoder_decoder,
        label2id=label2id,
        mode="head_tail_label",
        loop_dummy_relation_name=loop_dummy_relation_name,
        none_label=none_label,
    )

    if loop_dummy_relation_name is not None:
        with pytest.raises(ValueError) as excinfo:
            encoder_decoder.encode(
                BinaryRelation(
                    head=LabeledSpan(start=1, end=2, label="A"),
                    tail=LabeledSpan(start=1, end=2, label="A"),
                    label=loop_dummy_relation_name,
                )
            )

        assert (
            str(excinfo.value)
            == "loop_dummy_relation_name is set, but none_label is not set: None"
        )
    elif none_label is not None:
        none_id = label2id[none_label]
        with pytest.raises(ValueError) as excinfo:
            encoder_decoder.decode([4, 5, 1, none_id, none_id, none_id, none_id])
        assert (
            str(excinfo.value)
            == "loop_dummy_relation_name is not set, but none_label=N was found in decoded encoding: "
            "[4, 5, 1, 2, 2, 2, 2] (label2id: {'A': 0, 'B': 1, 'N': 2}))"
        )
    else:
        raise ValueError("unknown setting")


@pytest.mark.parametrize(
    "loop_dummy_relation_name,none_label",
    [(None, None), ("L", "N")],
)
def test_binary_relation_encoder_decoder_unknown_mode(loop_dummy_relation_name, none_label):
    """Test the BinaryRelationEncoderDecoder class."""

    label2id = {"A": 0, "B": 1, "N": 2, "L": 3}
    labeled_span_encoder_decoder = LabeledSpanEncoderDecoder(
        span_encoder_decoder=SpanEncoderDecoderWithOffset(offset=len(label2id)),
        label2id=label2id,
        mode="indices_label",
    )
    encoder_decoder = BinaryRelationEncoderDecoder(
        head_encoder_decoder=labeled_span_encoder_decoder,
        tail_encoder_decoder=labeled_span_encoder_decoder,
        label2id=label2id,
        mode="unknown",
        loop_dummy_relation_name=loop_dummy_relation_name,
        none_label=none_label,
    )
    with pytest.raises(ValueError) as excinfo:
        encoder_decoder.encode(
            BinaryRelation(
                head=LabeledSpan(start=1, end=2, label="A"),
                tail=LabeledSpan(start=1, end=2, label="A"),
                label="L",
            )
        )
    assert str(excinfo.value) == "unknown mode: unknown"

    with pytest.raises(ValueError) as excinfo:
        encoder_decoder.decode([2, 2, 2, 2, 2, 2, 2])
    assert str(excinfo.value) == "unknown mode: unknown"


def test_binary_relation_encoder_decoder_wrong_encoding_size():
    """Test the BinaryRelationEncoderDecoder class."""

    label2id = {"A": 0, "B": 1, "C": 2}
    labeled_span_encoder_decoder = LabeledSpanEncoderDecoder(
        span_encoder_decoder=SpanEncoderDecoderWithOffset(offset=len(label2id)),
        label2id=label2id,
        mode="indices_label",
    )
    encoder_decoder = BinaryRelationEncoderDecoder(
        head_encoder_decoder=labeled_span_encoder_decoder,
        tail_encoder_decoder=labeled_span_encoder_decoder,
        label2id=label2id,
        mode="head_tail_label",
    )
    with pytest.raises(DecodingLengthException) as excinfo:
        encoder_decoder.decode([1, 2, 3, 4, 5, 6])
    assert (
        str(excinfo.value)
        == "seven values are required to decode as BinaryRelation, but the encoding has length 6"
    )
    assert excinfo.value.identifier == "len"

    with pytest.raises(DecodingLengthException) as excinfo:
        encoder_decoder.decode([1, 2, 3, 4, 5, 6, 7, 8])
    assert (
        str(excinfo.value)
        == "seven values are required to decode as BinaryRelation, but the encoding has length 8"
    )
    assert excinfo.value.identifier == "len"


def test_binary_relation_encoder_decoder_wrong_label_index():
    """Test the BinaryRelationEncoderDecoder class."""

    label2id = {"A": 0, "B": 1, "C": 2}
    labeled_span_encoder_decoder = LabeledSpanEncoderDecoder(
        span_encoder_decoder=SpanEncoderDecoderWithOffset(offset=len(label2id)),
        label2id=label2id,
        mode="indices_label",
    )
    encoder_decoder = BinaryRelationEncoderDecoder(
        head_encoder_decoder=labeled_span_encoder_decoder,
        tail_encoder_decoder=labeled_span_encoder_decoder,
        label2id=label2id,
        mode="head_tail_label",
    )
    with pytest.raises(DecodingLabelException) as excinfo:
        encoder_decoder.decode([1, 2, 3, 4, 5, 6, 7])
    assert str(excinfo.value) == "unknown label id: 7 (label2id: {'A': 0, 'B': 1, 'C': 2})"
    assert excinfo.value.identifier == "label"
