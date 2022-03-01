from pytorch_ie.utils.window import enumerate_windows, get_window_around_slice


def test_enumerate_windows():
    sequence = [
        "Jane",
        "lives",
        "in",
        "Berlin",
        ".",
        "this",
        "is",
        "no",
        "sentence",
        "about",
        "Karl",
    ]
    windows = list(enumerate_windows(sequence=sequence, max_size=5))
    assert len(windows) == 3
    token_slice, label_slice = windows[0]
    assert token_slice == (0, 5)
    assert label_slice == (0, 5)

    token_slice, label_slice = windows[1]
    assert token_slice == (5, 10)
    assert label_slice == (0, 5)

    token_slice, label_slice = windows[2]
    assert token_slice == (10, 11)
    assert label_slice == (0, 1)


def test_enumerate_windows_with_overlap():
    sequence = [
        "Jane",
        "lives",
        "in",
        "Berlin",
        ".",
        "this",
        "is",
        "no",
        "sentence",
        "about",
        "Karl",
    ]
    windows = list(enumerate_windows(sequence=sequence, max_size=7, overlap=2))
    assert len(windows) == 3
    token_slice, label_slice = windows[0]
    assert token_slice == (0, 7)
    assert label_slice == (0, 5)

    token_slice, label_slice = windows[1]
    assert token_slice == (3, 10)
    assert label_slice == (2, 5)

    token_slice, label_slice = windows[2]
    assert token_slice == (6, 11)
    assert label_slice == (2, 5)


def test_enumerate_windows_with_overlap2():
    sequence = [
        "Seattle",
        "is",
        "a",
        "rainy",
        "city",
        ".",
        "Jenny",
        "Du",
        "##rka",
        "##n",
        "is",
        "the",
        "city",
        "'",
        "s",
        "mayor",
        ".",
    ]
    windows = list(enumerate_windows(sequence=sequence, max_size=14, overlap=3))
    assert len(windows) == 2
    token_slice, label_slice = windows[0]
    assert token_slice == (0, 14)
    assert label_slice == (0, 11)

    token_slice, label_slice = windows[1]
    assert token_slice == (8, 17)
    assert label_slice == (3, 9)


def test_get_window_around_slice():

    # default: result is centered around slice
    window_slice = get_window_around_slice(
        slice=(5, 7), max_window_size=6, available_input_length=10
    )
    assert window_slice == (3, 9)

    # slice at the beginning -> shift window to the right (regarding the slice center)
    window_slice = get_window_around_slice(
        slice=(0, 5), max_window_size=8, available_input_length=10
    )
    assert window_slice == (0, 8)

    # slice at the end -> shift window to the left (regarding the slice center)
    window_slice = get_window_around_slice(
        slice=(7, 10), max_window_size=8, available_input_length=10
    )
    assert window_slice == (2, 10)

    # max window size bigger than available_input_length -> take everything
    window_slice = get_window_around_slice(
        slice=(2, 6), max_window_size=8, available_input_length=7
    )
    assert window_slice == (0, 7)

    # slice exceeds max_window_size
    window_slice = get_window_around_slice(
        slice=(0, 5), max_window_size=4, available_input_length=10
    )
    assert window_slice is None
