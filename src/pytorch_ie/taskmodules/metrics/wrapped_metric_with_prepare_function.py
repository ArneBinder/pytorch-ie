import logging
from collections.abc import Collection, Sized
from typing import Any, Callable, Dict, Generic, Iterable, List, Optional, Tuple, TypeVar, Union

from torch import Tensor
from torchmetrics import Metric, MetricCollection
from torchmetrics.wrappers.abstract import WrapperMetric

logger = logging.getLogger(__name__)

T = TypeVar("T")
T2 = TypeVar("T2")


class WrappedMetricWithPrepareFunction(WrapperMetric, Generic[T]):
    """A wrapper around a metric that can be used with predictions and targets that are need to be
    prepared (e.g. un-batched) before passing them to the metric.

    Args:
        metric: The metric to wrap. It should be a subclass of torchmetrics.Metric.
        prepare_function: A function that prepares the input for the metric. If provided, It is called with
            the predictions as well as the targets (separately).
        prepare_together_function: A function that prepares both the predictions and the targets together and
            should return them as a tuple. If provided, it is called with the predictions and the targets as
            arguments.
        prepare_does_unbatch: If True, the prepare_function is expected to return an iterable of
            individual inputs. This can be used to un-batch the input before passing it to the
            wrapped metric.
    """

    def __init__(
        self,
        metric: Union[Metric, MetricCollection],
        prepare_function: Optional[Callable[[T], Any]] = None,
        prepare_together_function: Optional[Callable[[T, T], Tuple[Any, Any]]] = None,
        prepare_does_unbatch: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.metric = metric
        self.prepare_function = prepare_function
        self.prepare_both_function = prepare_together_function
        self.prepare_does_unbatch = prepare_does_unbatch

    def _is_empty_batch(self, prediction: T2, target: T2) -> bool:
        if isinstance(prediction, Sized) and isinstance(target, Sized):
            pred_len = len(prediction)
            target_len = len(target)
        else:
            raise ValueError(
                "Both prediction and target need to be sized when prepare_does_unbatch=False."
            )
        if pred_len != target_len:
            raise ValueError(
                f"Number of elements in prediction ({pred_len}) and target ({target_len}) do not match."
            )
        if pred_len == 0:
            return True
        return False

    def forward(self, prediction: T, target: T) -> Any:
        if self.prepare_function is not None:
            prediction = self.prepare_function(prediction)
            target = self.prepare_function(target)
        if self.prepare_both_function is not None:
            prediction, target = self.prepare_both_function(prediction, target)
        if self.prepare_does_unbatch:
            if not isinstance(prediction, Collection) or not isinstance(target, Collection):
                raise ValueError(
                    "Both prediction and target need to be iterable and sized when prepare_does_unbatch=True."
                )
            if len(prediction) != len(target):
                raise ValueError(
                    f"Number of prepared predictions ({len(prediction)}) and targets "
                    f"({len(target)}) do not match."
                )
            if len(prediction) == 0:
                raise ValueError("Empty batch.")
            results = []
            for prediction_str, target_str in zip(prediction, target):
                current_result = self.metric(prediction_str, target_str)
                results.append(current_result)
            return results
        else:
            if not self._is_empty_batch(prediction, target):
                return self.metric(prediction, target)
            else:
                return None

    def update(self, prediction: T, target: T) -> None:
        if self.prepare_function is not None:
            prediction = self.prepare_function(prediction)
            target = self.prepare_function(target)
        if self.prepare_both_function is not None:
            prediction, target = self.prepare_both_function(prediction, target)
        if self.prepare_does_unbatch:
            if not isinstance(prediction, Collection) or not isinstance(target, Collection):
                raise ValueError(
                    "Both prediction and target need to be iterable and sized when prepare_does_unbatch=True."
                )
            if len(prediction) != len(target):
                raise ValueError(
                    f"Number of prepared predictions ({len(prediction)}) and targets "
                    f"({len(target)}) do not match."
                )
            if len(prediction) == 0:
                raise ValueError("Empty batch.")
            for prediction_str, target_str in zip(prediction, target):
                self.metric.update(prediction_str, target_str)
        else:
            if not self._is_empty_batch(prediction, target):
                self.metric.update(prediction, target)

    def compute(self) -> Any:
        return self.metric.compute()

    def reset(self) -> None:
        self.metric.reset()

    @property
    def metric_state(self) -> Dict[str, Union[List[Tensor], Tensor]]:
        if isinstance(self.metric, Metric):
            return self.metric.metric_state
        elif isinstance(self.metric, MetricCollection):
            # TODO: maypy complains. is the code wrong or mypy?
            result = {
                metric_name: metric.metric_state for metric_name, metric in self.metric.items()
            }
            return result  # type: ignore[return-value]
        else:
            raise ValueError(f"Unsupported metric type: {type(self.metric)}")
