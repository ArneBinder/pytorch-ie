import logging
from typing import Any, Callable, Dict, Generic, Optional, Sequence, Tuple, TypeVar

import torch
from torch.nn import ModuleDict
from torchmetrics import Metric

from .common import MetricWithArbitraryCounts

logger = logging.getLogger(__name__)
T = TypeVar("T")
U = TypeVar("U")


class WrappedLayerMetricsWithUnbatchAndDecodeWithErrorsFunction(
    MetricWithArbitraryCounts, Generic[T, U]
):
    """A wrapper around annotation layer metrics that can be used with batched encoded annotations.

    Args:
        layer_metrics: A dictionary mapping layer names to annotation layer metrics. Each metric
            should be a subclass of torchmetrics.Metric and should take two sets of annotations as
            input.
        unbatch_function: A function that takes a batched input and returns an iterable of
            individual inputs. This is used to unbatch the input before passing it to the annotation
            decoding function (decode_annotations_with_errors_function).
        decode_layers_with_errors_function: A function that takes an annotation encoding and
            returns a tuple of two dictionaries. The first dictionary maps layer names to a list of
            annotations. The second dictionary maps error names to the number of errors that were
            encountered while decoding the annotations.
        round_precision: The number of digits to round the results to. If None, no rounding is
            performed.
        error_key_correct: The key in the error dictionary whose value should be the number of *correctly*
            decoded annotations, so that the sum of all values in the error dictionary can be used to
            normalize the error counts. If None, the total number of training examples is used to
            normalize the error counts.
        collect_exact_encoding_matches: Whether to collect the number of examples where the full target encoding
            was predicted correctly (exact matches).
    """

    def __init__(
        self,
        layer_metrics: Dict[str, Metric],
        unbatch_function: Callable[[T], Sequence[U]],
        decode_layers_with_errors_function: Callable[[U], Tuple[Dict[str, Any], Dict[str, int]]],
        round_precision: Optional[int] = 4,
        error_key_correct: Optional[str] = None,
        collect_exact_encoding_matches: bool = True,
    ):
        super().__init__()

        self.key_error_correct = error_key_correct
        self.collect_exact_encoding_matches = collect_exact_encoding_matches
        self.round_precision = round_precision
        self.unbatch_function = unbatch_function
        self.decode_layers_with_errors_function = decode_layers_with_errors_function
        self.layer_metrics = ModuleDict(layer_metrics)

        # total number of encodings
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        # this contains the number of examples where the full target sequence was predicted correctly (exact matches)
        self.add_state("exact_encoding_matches", default=torch.tensor(0), dist_reduce_fx="sum")
        # note: the error counts are stored via the MetricWithArbitraryCounts base class

    def update(self, prediction, expected):
        prediction_list = self.unbatch_function(prediction)
        expected_list = self.unbatch_function(expected)
        if len(prediction_list) != len(expected_list):
            raise ValueError(
                f"Number of predictions ({len(prediction_list)}) and targets ({len(expected_list)}) do not match."
            )

        for expected_encoding, prediction_encoding in zip(expected_list, prediction_list):
            expected_layers, _ = self.decode_layers_with_errors_function(expected_encoding)
            predicted_layers, predicted_errors = self.decode_layers_with_errors_function(
                prediction_encoding
            )
            for k, v in predicted_errors.items():
                self.inc_counts(counts=torch.tensor(v).to(self.device), key=k, prefix="errors_")

            for layer_name, metric in self.layer_metrics.items():
                metric.update(expected_layers[layer_name], predicted_layers[layer_name])

            if self.collect_exact_encoding_matches:
                if isinstance(expected_encoding, torch.Tensor) and isinstance(
                    prediction_encoding, torch.Tensor
                ):
                    is_match = torch.equal(expected_encoding, prediction_encoding)
                else:
                    is_match = expected_encoding == prediction_encoding
                if is_match:
                    self.exact_encoding_matches += 1

            self.total += 1

    def reset(self):
        super().reset()

        for metric in self.layer_metrics.values():
            metric.reset()

    def _nested_round(self, d: Dict[str, Any]) -> Dict[str, Any]:
        if self.round_precision is None:
            return d
        res: Dict[str, Any] = {}
        for k, v in d.items():
            if isinstance(v, dict):
                res[k] = self._nested_round(v)
            elif isinstance(v, float):
                res[k] = round(v, self.round_precision)
            else:
                res[k] = v
        return res

    def compute(self):
        res = {}
        if self.collect_exact_encoding_matches:
            res["exact_encoding_matches"] = (
                self.exact_encoding_matches / self.total if self.total > 0 else 0.0
            )

        errors = self.get_counts(key_prefix="errors_")
        # if errors contains a "correct" key, use that to normalize, otherwise use the number of training examples
        if self.key_error_correct in errors:
            errors_total = sum(errors.values())
        else:
            errors_total = self.total
        res["decoding_errors"] = {
            k: v / errors_total if errors_total > 0 else 0.0 for k, v in errors.items()
        }
        if "all" not in res["decoding_errors"]:
            res["decoding_errors"]["all"] = (
                sum(v for k, v in errors.items() if k != self.key_error_correct) / errors_total
                if errors_total > 0
                else 0.0
            )

        for layer_name, metric in self.layer_metrics.items():
            if layer_name in res:
                raise ValueError(
                    f"Layer name '{layer_name}' is already used in the metric result dictionary."
                )
            res[layer_name] = metric.compute()

        res = self._nested_round(res)

        return res
