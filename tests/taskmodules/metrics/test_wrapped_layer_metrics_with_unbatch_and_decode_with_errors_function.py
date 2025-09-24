import json
from typing import Any, Dict, Tuple

import pytest
from torch import tensor
from torchmetrics import Metric

from pytorch_ie.taskmodules.metrics import (
    WrappedLayerMetricsWithUnbatchAndDecodeWithErrorsFunction,
)


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


@pytest.fixture(scope="module")
def wrapped_layer_metrics_with_unbatch_and_decode_with_errors_function():
    def decode_with_errors_function(x: str) -> Tuple[Dict[str, Any], Dict[str, int]]:
        if x == "error":
            return {"entities": [], "relations": []}, {"dummy": 1}
        else:
            return json.loads(x), {"dummy": 0}

    layer_metrics = {
        "entities": TestMetric(),
        "relations": TestMetric(),
    }
    metric = WrappedLayerMetricsWithUnbatchAndDecodeWithErrorsFunction(
        layer_metrics=layer_metrics,
        unbatch_function=lambda x: x.split("\n"),
        decode_layers_with_errors_function=decode_with_errors_function,
    )
    return metric


def test_wrapped_layer_metrics_with_unbatch_and_decode_with_errors_function(
    wrapped_layer_metrics_with_unbatch_and_decode_with_errors_function,
):
    metric = wrapped_layer_metrics_with_unbatch_and_decode_with_errors_function
    assert metric is not None
    assert metric.unbatch_function is not None
    assert metric.decode_layers_with_errors_function is not None
    assert metric.layer_metrics is not None
    assert metric.metric_state == {
        "total": tensor(0),
        "exact_encoding_matches": tensor(0),
    }

    values = metric.compute()
    assert metric.metric_state
    assert values == {
        "decoding_errors": {"all": 0.0},
        "entities": 0.0,
        "exact_encoding_matches": 0.0,
        "relations": 0.0,
    }

    metric.reset()
    # Prediction and expected are the same.
    metric.update(
        prediction=json.dumps({"entities": ["E1"], "relations": ["R1"]}),
        expected=json.dumps({"entities": ["E1"], "relations": ["R1"]}),
    )
    assert metric.metric_state == {
        "total": tensor(1),
        "exact_encoding_matches": tensor(1),
        "errors_dummy": tensor(0),
    }
    values = metric.compute()
    assert values == {
        "decoding_errors": {"all": 0.0, "dummy": 0.0},
        "entities": 1.0,
        "exact_encoding_matches": 1.0,
        "relations": 1.0,
    }

    metric.reset()
    # Prediction and expected are different and there are multiple entries.
    # The first entry is an exact match, the second entry is not.
    metric.update(
        prediction=json.dumps({"entities": ["E1"], "relations": ["R1"]})
        + "\n"
        + json.dumps({"entities": ["E1"], "relations": ["R1"]}),
        expected=json.dumps({"entities": ["E1"], "relations": ["R1"]})
        + "\n"
        + json.dumps({"entities": ["E1"], "relations": ["R2"]}),
    )
    assert metric.metric_state == {
        "total": tensor(2),
        "exact_encoding_matches": tensor(1),
        "errors_dummy": tensor(0),
    }
    values = metric.compute()
    assert values == {
        "decoding_errors": {"all": 0.0, "dummy": 0.0},
        "entities": 1.0,
        "exact_encoding_matches": 0.5,
        "relations": 0.5,
    }

    metric.reset()
    # Encoding error
    metric.update(
        prediction="error",
        expected=json.dumps({"entities": ["E1"], "relations": []}),
    )
    assert metric.metric_state == {
        "total": tensor(1),
        "exact_encoding_matches": tensor(0),
        "errors_dummy": tensor(1),
    }
    values = metric.compute()
    # In the case on an error, the decoding function returns adict with empty lists for entities and relations.
    # Thus, we get a perfect match for entities and a 0.0 match for relations.
    assert values == {
        "decoding_errors": {"all": 1.0, "dummy": 1.0},
        "entities": 0.0,
        "exact_encoding_matches": 0.0,
        "relations": 1.0,
    }

    # test mismatched number of predictions and targets
    metric.reset()
    with pytest.raises(ValueError) as excinfo:
        metric.update(
            prediction=json.dumps({"entities": ["E1"], "relations": ["R1"]}),
            expected=json.dumps({"entities": ["E1"], "relations": ["R1"]})
            + "\n"
            + json.dumps({"entities": ["E1"], "relations": ["R1"]}),
        )
    assert str(excinfo.value) == "Number of predictions (1) and targets (2) do not match."
