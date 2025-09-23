import pytest
from pie_documents.annotations import LabeledSpan
from torch import tensor

from pytorch_ie.taskmodules.metrics import PrecisionRecallAndF1ForLabeledAnnotations


def test_precision_recall_and_f1_for_labeled_annotations():
    metric = PrecisionRecallAndF1ForLabeledAnnotations()
    assert metric.metric_state == {}

    metric.update(
        gold=[LabeledSpan(start=0, end=1, label="a")],
        predicted=[LabeledSpan(start=0, end=1, label="a")],
    )
    metric_state = {k: v.tolist() for k, v in metric.metric_state.items()}
    assert metric_state == {"counts_a": [1, 1, 1], "counts_micro": [1, 1, 1]}
    value = metric.compute()
    assert value == {
        "a": {"recall": 1.0, "precision": 1.0, "f1": 1.0},
        "macro": {"recall": 1.0, "precision": 1.0, "f1": 1.0},
        "micro": {"recall": 1.0, "precision": 1.0, "f1": 1.0},
    }

    metric.reset()
    metric.update(
        gold=[LabeledSpan(start=0, end=1, label="a"), LabeledSpan(start=0, end=1, label="b")],
        predicted=[LabeledSpan(start=0, end=1, label="b"), LabeledSpan(start=0, end=1, label="c")],
    )
    metric_state = {k: v.tolist() for k, v in metric.metric_state.items()}
    assert metric_state == {
        "counts_a": [1, 0, 0],
        "counts_b": [1, 1, 1],
        "counts_c": [0, 1, 0],
        "counts_micro": [2, 2, 1],
    }
    assert metric.compute() == {
        "b": {"recall": 1.0, "precision": 1.0, "f1": 1.0},
        "a": {"recall": 0.0, "precision": 0.0, "f1": 0.0},
        "c": {"recall": 0.0, "precision": 0.0, "f1": 0.0},
        "macro": {
            "f1": tensor(0.3333333432674408),
            "precision": tensor(0.3333333432674408),
            "recall": tensor(0.3333333432674408),
        },
        "micro": {"recall": 0.5, "precision": 0.5, "f1": 0.5},
    }

    # check deduplication in same update
    metric.reset()
    metric.update(
        gold=[
            LabeledSpan(start=0, end=1, label="a"),
            LabeledSpan(start=0, end=1, label="a"),
            LabeledSpan(start=0, end=1, label="b"),
        ],
        predicted=[
            LabeledSpan(start=0, end=1, label="b"),
            LabeledSpan(start=0, end=1, label="b"),
            LabeledSpan(start=0, end=1, label="c"),
        ],
    )
    metric_state = {k: v.tolist() for k, v in metric.metric_state.items()}
    assert metric_state == {
        "counts_a": [1, 0, 0],
        "counts_b": [1, 1, 1],
        "counts_c": [0, 1, 0],
        "counts_micro": [2, 2, 1],
    }

    # assert no deduplication over multiple updates
    metric.reset()
    metric.update(
        gold=[LabeledSpan(start=0, end=1, label="a")],
        predicted=[LabeledSpan(start=0, end=1, label="b")],
    )
    metric.update(
        gold=[LabeledSpan(start=0, end=1, label="b")],
        predicted=[LabeledSpan(start=0, end=1, label="a")],
    )
    metric_state = {k: v.tolist() for k, v in metric.metric_state.items()}
    assert metric_state == {
        "counts_a": [1, 1, 0],
        "counts_b": [1, 1, 0],
        "counts_c": [0, 0, 0],
        "counts_micro": [2, 2, 0],
    }
    assert metric.compute() == {
        "a": {"f1": 0.0, "precision": 0.0, "recall": 0.0},
        "b": {"f1": 0.0, "precision": 0.0, "recall": 0.0},
        "c": {"f1": 0.0, "precision": 0.0, "recall": 0.0},
        "macro": {"f1": 0.0, "precision": 0.0, "recall": 0.0},
        "micro": {"f1": 0.0, "precision": 0.0, "recall": 0.0},
    }


def test_precision_recall_and_f1_for_labeled_annotations_in_percent():
    metric = PrecisionRecallAndF1ForLabeledAnnotations(
        in_percent=True, flatten_result_with_sep="/"
    )

    metric.update(
        gold=[LabeledSpan(start=0, end=1, label="a")],
        predicted=[LabeledSpan(start=0, end=1, label="a"), LabeledSpan(start=0, end=1, label="b")],
    )
    values = {k: v.item() for k, v in metric.compute().items()}
    assert values == {
        "a/f1": 100.0,
        "a/precision": 100.0,
        "a/recall": 100.0,
        "b/f1": 0.0,
        "b/precision": 0.0,
        "b/recall": 0.0,
        "macro/f1": 50.0,
        "macro/precision": 50.0,
        "macro/recall": 50.0,
        "micro/f1": 66.66667175292969,
        "micro/precision": 50.0,
        "micro/recall": 100.0,
    }


def test_precision_recall_and_f1_for_labeled_annotations_with_label_mapping():
    metric = PrecisionRecallAndF1ForLabeledAnnotations(
        label_mapping={"a": "label_a", "b": "label_b"}
    )

    metric.update(
        gold=[LabeledSpan(start=0, end=1, label="a")],
        predicted=[LabeledSpan(start=0, end=1, label="a"), LabeledSpan(start=0, end=1, label="b")],
    )
    assert metric.compute() == {
        "label_a": {"f1": 1.0, "precision": 1.0, "recall": 1.0},
        "label_b": {"f1": 0.0, "precision": 0.0, "recall": 0.0},
        "macro": {"f1": 0.5, "precision": 0.5, "recall": 0.5},
        "micro": {"f1": 0.6666666666666666, "precision": 0.5, "recall": 1.0},
    }


def test_precision_recall_and_f1_for_labeled_annotations_key_micro_error():
    metric = PrecisionRecallAndF1ForLabeledAnnotations()
    with pytest.raises(ValueError) as excinfo:
        metric.update(
            gold=[LabeledSpan(start=0, end=1, label="micro")],
            predicted=[],
        )
    assert (
        str(excinfo.value)
        == "The key 'micro' was used as an annotation label, but it is reserved for the micro average. "
        "You can change which key is used for that with the 'key_micro' argument."
    )
