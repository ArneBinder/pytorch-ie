from typing import Iterator, Optional, Sequence, Tuple


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


def get_window_around_slice(
    slice: Tuple[int, int], max_window_size: int, available_input_length: int
) -> Optional[Tuple[int, int]]:
    """
    Given a `max_window` size, `available_token_length` and a `slice` (pair of start and end indices) that
    is required to be in the resulting window, create a new slice of size `max_window_size` (or less, if not possible)
    around the required slice. Per default, the resulting slice will be centered around the required slice.
    However, if the required slice is at the beginning or end of the available tokens, the resulting window is
    shifted to contain as many tokens as possible.
    Iff the required `slice` already exceeds the `max_window_size`, return `None`.
    """

    # current pair may not fit into the window
    if slice[1] - slice[0] > max_window_size:
        return None

    # set the final window size (regarding input tokens)
    window_size = min(available_input_length, max_window_size)

    rel_center = sum(slice) / 2.0
    window_start = int(rel_center - window_size / 2.0)
    window_end = window_start + window_size

    # If window goes over one end, shift it use as much content as possible.
    # First shift window to left and then to right to ensure that window_start is never
    # negative (if window_end is outside, this will not be a problem)
    if window_end >= available_input_length:
        delta = available_input_length - window_end
        window_start += delta
        window_end += delta
    if window_start < 0:
        delta = -window_start
        window_start += delta
        window_end += delta
    assert (
        0 <= window_start < available_input_length
    ), f"window_start={window_start} not available in sequence"

    return window_start, window_end
