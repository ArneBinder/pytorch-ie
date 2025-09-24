import logging
from typing import Any, Dict, Generic, Mapping, Tuple, TypeVar

from torch import Tensor
from typing_extensions import TypeAlias

from .model_with_metrics_from_taskmodule import ModelWithMetricsFromTaskModule
from .stages import TESTING, TRAINING, VALIDATION

InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")
TargetType = TypeVar("TargetType")
StepInputType: TypeAlias = Tuple[InputType, TargetType]
# bound is defined by lightning's training/test_step return type
StepOutputType = TypeVar("StepOutputType", bound=Tensor | Mapping[str, Tensor] | None)

logger = logging.getLogger(__name__)


class ModelWithBoilerplate(
    ModelWithMetricsFromTaskModule[InputType, TargetType, OutputType],
    Generic[InputType, OutputType, TargetType, StepOutputType],
):
    """A PyTorchIEModel that adds boilerplate code for training, validation, and testing.

    Especially, it handles updating the metrics and logging of losses and metric results. Also see
    ModelWithMetricsFromTaskModule for more details on how metrics are handled.
    """

    def get_loss_from_outputs(self, outputs: OutputType) -> StepOutputType:
        if hasattr(outputs, "loss"):
            return outputs.loss
        else:
            raise ValueError(
                f"The model {self.__class__.__name__} does not define a 'loss' attribute in its output, "
                "so the loss cannot be automatically extracted from the outputs. Please override the"
                "get_loss_from_outputs() method for this model."
            )

    def log_loss(self, stage: str, loss: StepOutputType) -> None:
        # show loss on each step only during training
        log_kwargs: Dict[str, Any] = dict(
            on_step=(stage == TRAINING), on_epoch=True, prog_bar=True, sync_dist=True
        )
        if isinstance(loss, Tensor):
            self.log(f"loss/{stage}", loss, **log_kwargs)
        elif isinstance(loss, Mapping):
            for k, v in loss.items():
                if isinstance(v, Tensor):
                    self.log(f"loss/{k}/{stage}", v, **log_kwargs)
                else:
                    logger.warning(
                        f"The loss item '{k}' is of type {type(v)}, but only Tensor losses can be logged."
                    )

    def _step(self, stage: str, batch: StepInputType) -> StepOutputType:
        inputs, targets = batch
        outputs = self(inputs=inputs, targets=targets)
        loss = self.get_loss_from_outputs(outputs=outputs)
        self.log_loss(stage=stage, loss=loss)
        self.update_metric(inputs=inputs, outputs=outputs, targets=targets, stage=stage)

        return loss

    def training_step(self, batch: StepInputType, batch_idx: int) -> StepOutputType:
        return self._step(stage=TRAINING, batch=batch)

    def validation_step(self, batch: StepInputType, batch_idx: int) -> StepOutputType:
        return self._step(stage=VALIDATION, batch=batch)

    def test_step(
        self, batch: StepInputType, batch_idx: int, *args: Any, **kwargs: Any
    ) -> StepOutputType:
        return self._step(stage=TESTING, batch=batch)

    def predict_step(
        self, batch: StepInputType, batch_idx: int, dataloader_idx: int = 0
    ) -> TargetType:
        inputs, targets = batch
        return self.predict(inputs=inputs)

    def on_train_epoch_end(self) -> None:
        self.log_metric(stage=TRAINING)

    def on_validation_epoch_end(self) -> None:
        self.log_metric(stage=VALIDATION)

    def on_test_epoch_end(self) -> None:
        self.log_metric(stage=TESTING)
