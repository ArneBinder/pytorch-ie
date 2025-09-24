from functools import partial

import pytest
from torchmetrics import Metric

from pytorch_ie.taskmodules.metrics import WrappedMetricWithPrepareFunction


class TestMetric(Metric):
    """A simple metric that computes the exact match ratio between predictions and targets."""

    def __init__(self):
        super().__init__()
        self.add_state("matching", default=[])

    def update(self, prediction: str, target: str):
        self.matching.append(prediction == target)

    def compute(self):
        # Note: returning NaN in the case of an empty list would be more correct, but
        #   returning 0.0 is more convenient for testing.
        return sum(self.matching) / len(self.matching) if self.matching else 0.0


def test_metric():
    metric = WrappedMetricWithPrepareFunction(
        metric=TestMetric(), prepare_function=lambda x: x.split()[0]
    )

    assert metric is not None
    assert metric.prepare_function is not None

    assert metric.compute() == 0.0

    metric.reset()
    metric(prediction="abc", target="abc")
    assert metric.compute() == 1.0

    metric.reset()
    metric(prediction="abc", target="def")
    assert metric.compute() == 0.0

    metric.reset()
    metric(prediction="abc def", target="abc xyz")
    # we consider just the first word, so this is still 1.0
    assert metric.compute() == 1.0

    metric.reset()
    metric(prediction="abc def", target="xyz def")
    assert metric.compute() == 0.0


def split_both_and_remove_where_both_match(
    preds: str, targets: str, match: str
) -> tuple[list[str], list[str]]:
    preds_list = preds.split()
    targets_list = targets.split()
    not_both_none_indices = [
        i for i, (p, t) in enumerate(zip(preds_list, targets_list)) if p != match or t != match
    ]
    preds_list = [preds_list[i] for i in not_both_none_indices]
    targets_list = [targets_list[i] for i in not_both_none_indices]
    return preds_list, targets_list


def test_wrapped_metric_with_prepare_both_function():
    metric = WrappedMetricWithPrepareFunction(
        metric=TestMetric(),
        prepare_together_function=partial(split_both_and_remove_where_both_match, match="none"),
        prepare_does_unbatch=True,
    )

    assert metric is not None
    assert metric.prepare_both_function is not None

    assert metric.compute() == 0.0

    # none is removed from both, remaining is the same
    metric.reset()
    metric(prediction="abc none", target="abc none")
    assert metric.compute() == 1.0

    # none is removed from both, remaining is different
    metric.reset()
    metric(prediction="abc none", target="def none")
    assert metric.compute() == 0.0

    # none is not removed from both, remaining is partially the same
    metric.reset()
    metric(prediction="abc def", target="abc none")
    assert metric.compute() == 0.5

    # none is not removed from both, remaining is different
    metric.reset()
    metric(prediction="abc def", target="def none")
    assert metric.compute() == 0.0


@pytest.fixture(scope="module")
def wrapped_metric_with_unbatch_function():
    # just split the strings to unbatch the inputs
    return WrappedMetricWithPrepareFunction(
        metric=TestMetric(), prepare_function=lambda x: x.split(), prepare_does_unbatch=True
    )


def test_wrapped_metric_with_unbatch_function(wrapped_metric_with_unbatch_function):
    metric = wrapped_metric_with_unbatch_function
    assert metric is not None

    assert metric.compute() == 0.0

    metric.reset()
    metric(prediction="abc", target="abc")
    assert metric.compute() == 1.0

    metric.reset()
    metric(prediction="abc", target="def")
    assert metric.compute() == 0.0

    metric.reset()
    metric(prediction="abc def", target="abc def")
    assert metric.compute() == 1.0

    metric.reset()
    metric(prediction="abc def", target="def abc")
    assert metric.compute() == 0.0

    metric.reset()
    metric(prediction="abc xyz", target="def xyz")
    assert metric.compute() == 0.5


def test_wrapped_metric_with_unbatch_function_size_mismatch(wrapped_metric_with_unbatch_function):
    with pytest.raises(ValueError) as excinfo:
        wrapped_metric_with_unbatch_function(prediction="abc", target="abc def")
    assert str(excinfo.value) == "Number of prepared predictions (1) and targets (2) do not match."
