import logging
from typing import Any, Dict, Generic, List, Optional, Set, TypeVar, Union

from pie_core.utils.dictionary import flatten_dict_s
from torch import Tensor
from torchmetrics import Metric, MetricCollection

from pytorch_ie import PyTorchIEModel

from .has_taskmodule import HasTaskmodule
from .stages import TESTING, TRAINING, VALIDATION

InputType = TypeVar("InputType")
TargetType = TypeVar("TargetType")
OutputType = TypeVar("OutputType")

# from torchmetrics
_METRIC = (Metric, Tensor, int, float)

logger = logging.getLogger(__name__)


class ModelWithMetricsFromTaskModule(
    HasTaskmodule, PyTorchIEModel, Generic[InputType, TargetType, OutputType]
):
    """A PyTorchIEModel that adds metrics from a taskmodule.

    The metrics are added to the model as attributes with the names metric_{stage} via
    setup_metrics method, where stage is one of "train", "val", or "test". The metrics are updated
    with the update_metric method and logged with the log_metric method.

    Args:
        metric_stages: The stages for which to set up metrics. Must be one of "train", "val", or
            "test".
        metric_intervals: A dict mapping metric stages to the number of steps between metric
            calculation. If not provided, the metrics are calculated at the end of each epoch.
        metric_call_predict: Whether to call predict() and use its result for metric calculation
            instead of the (decoded) model output. This is useful, for instance, for generative models
            that define special logic to produce predictions, e.g. beam search, which requires multiple
            passes through the model. If True, predict() is called for all metric stages. If False (default),
            the model outputs are passed to decode() and that is used for all metric stages. If a list of
            metric stages is provided, predict() is called for these stages and the (decoded) model
            outputs for the remaining stages.
    """

    def __init__(
        self,
        metric_stages: List[str] = [TRAINING, VALIDATION, TESTING],
        metric_intervals: Optional[Dict[str, int]] = None,
        metric_call_predict: Union[bool, List[str]] = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.setup_metrics(metric_stages=metric_stages)

        self.metric_intervals = metric_intervals or {}
        missed_stages = set(self.metric_intervals) - set(metric_stages)
        if len(missed_stages) > 0:
            logger.warning(
                f"There are stages in metric_intervals that are not in metric_stages: "
                f"{missed_stages}. Available metric stages: {metric_stages}."
            )

        self.use_prediction_for_metrics: Set[str]
        if isinstance(metric_call_predict, bool):
            self.metric_call_predict = set(metric_stages) if metric_call_predict else set()
        else:
            self.metric_call_predict = set(metric_call_predict)
        missed_stages = self.metric_call_predict - set(metric_stages)
        if len(missed_stages) > 0:
            logger.warning(
                f"There are stages in metric_call_predict that are not in metric_stages: "
                f"{missed_stages}. Available metric stages: {metric_stages}."
            )

    def setup_metrics(self, metric_stages: List[str]) -> None:
        """Set up metrics for the given stages if a taskmodule is available.

        Args:
            metric_stages: The stages for which to set up metrics. Must be one of "train", "val", or
                "test".
        """
        if self.taskmodule is not None:
            for stage in metric_stages:
                metric = self.taskmodule.configure_model_metric(stage=stage)
                if metric is not None:
                    self._set_metric(stage=stage, metric=metric)
                else:
                    logger.warning(
                        f"The taskmodule {self.taskmodule.__class__.__name__} does not define a metric for stage "
                        f"'{stage}'."
                    )
        elif len(metric_stages) > 0:
            logger.warning(
                "No taskmodule is available, so no metrics are set up. "
                "Please provide a taskmodule_config to enable metrics for stages "
                f"{metric_stages}."
            )

    def _get_metric(
        self, stage: str, batch_idx: int = 0
    ) -> Optional[Union[Metric, MetricCollection]]:
        metric_interval = self.metric_intervals.get(stage, 1)
        if (batch_idx + 1) % metric_interval == 0:
            return getattr(self, f"metric_{stage}", None)
        else:
            return None

    def _set_metric(self, stage: str, metric: Optional[Union[Metric, MetricCollection]]) -> None:
        setattr(self, f"metric_{stage}", metric)

    def update_metric(
        self,
        stage: str,
        inputs: InputType,
        targets: TargetType,
        outputs: OutputType,
    ) -> None:
        """Update the metric for the given stage. If outputs is provided, the predictions are
        decoded from the outputs. Otherwise, the predictions are obtained by directly calling the
        predict method with the inputs (note that this causes the model to be called a second
        time). Finally, the metric is updated with the predictions and targets.

        Args:
            stage: The stage for which to update the metric. Must be one of "train", "val", or "test".
            inputs: The inputs to the model.
            targets: The targets for the inputs.
            outputs: The outputs of the model. They are decoded into predictions if provided. If
                outputs is None, the predictions are obtained by directly calling the predict method
                on the inputs.
        """

        metric = self._get_metric(stage=stage)
        if metric is not None:
            if stage in self.metric_call_predict:
                predictions = self.predict(inputs=inputs)
            else:
                predictions = self.decode(inputs=inputs, outputs=outputs)
            metric.update(predictions, targets)

    def log_metric(self, stage: str, reset: bool = True) -> None:
        """Log the metric for the given stage and reset it."""

        metric = self._get_metric(stage=stage)
        if metric is not None:
            values = metric.compute()
            log_kwargs: Dict[str, Any] = {"on_step": False, "on_epoch": True, "sync_dist": True}
            if isinstance(values, dict):
                values_flat = flatten_dict_s(values, sep="/")
                for key, value in values_flat.items():
                    if isinstance(value, _METRIC):
                        self.log(f"metric/{key}/{stage}", value, **log_kwargs)
                    else:
                        raise ValueError(
                            f"Metric compute() returned an unsupported type for key '{key}': {type(value)}. "
                            "Expected a single value (int, float, or Tensor) or a torchmetrics.Metric."
                        )
            elif isinstance(values, _METRIC):
                metric_name = getattr(metric, "name", None) or type(metric).__name__
                self.log(f"metric/{metric_name}/{stage}", values, **log_kwargs)
            else:
                raise ValueError(
                    f"Metric compute() returned an unsupported type: {type(values)}. "
                    "Expected a dict or a single value (int, float, or Tensor) or a torchmetrics.Metric."
                )
            if reset:
                metric.reset()
