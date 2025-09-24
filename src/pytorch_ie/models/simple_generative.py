import copy
import logging
from typing import Any, Dict, Optional, Tuple, Type, Union

import torch
from pie_core import Model
from pie_core.utils.hydra import resolve_type
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import FloatTensor, LongTensor
from torch.optim import Optimizer
from transformers import PreTrainedModel, SchedulerType, get_scheduler
from transformers.modeling_outputs import Seq2SeqLMOutput
from typing_extensions import TypeAlias

from .common import ModelWithBoilerplate

logger = logging.getLogger(__name__)

# model inputs / outputs / targets
InputType: TypeAlias = Dict[str, LongTensor]
OutputType: TypeAlias = Seq2SeqLMOutput
TargetType: TypeAlias = Dict[str, LongTensor]
# step inputs (batch) / outputs (loss)
StepInputType: TypeAlias = Tuple[InputType, TargetType]
StepOutputType: TypeAlias = FloatTensor


@Model.register()
class SimpleGenerativeModel(
    ModelWithBoilerplate[InputType, OutputType, TargetType, StepOutputType],
):
    """This model is a simple wrapper around a generative model from Huggingface transformers. That
    means, its predict() and predict_step() methods will call the generate() method of the base
    model.

    If a taskmodule config is provided, the taskmodule will be instantiated and used to create metrics and
    a generation config with its configure_model_metric() and configure_model_generation() methods,
    respectively.

    If the base model has a configure_optimizer() method, this will be used to create the optimizer. Otherwise,
    the optimizer_type and learning_rate will be used to create an optimizer.

    Args:
        base_model_type: The type of the base model, e.g. "transformers.AutoModelForSeq2SeqLM". It should have a
            from_pretrained() method.
        base_model_config: A dictionary with the keyword arguments that will be passed to the from_pretrained()
            method of the base model.
        override_generation_kwargs: The generation config for the base model. This will override the generation config
            from the taskmodule, if one is provided.
        warmup_proportion: The proportion of the training steps that will be used for the warmup of the learning rate
            scheduler.
        learning_rate: The learning rate for the optimizer. If the base model has a configure_optimizer() method, this
            will be ignored.
        optimizer_type: The type of the optimizer. If the base model has a configure_optimizer() method, this will be
            ignored.
        **kwargs: Additional keyword arguments that will be passed to the PyTorchIEModel constructor.
    """

    def __init__(
        self,
        # base model setup
        base_model: Optional[Dict[str, Any]] = None,
        # old setup
        base_model_type: Optional[str] = None,
        base_model_config: Optional[Dict[str, Any]] = None,
        # generation
        override_generation_kwargs: Optional[Dict[str, Any]] = None,
        # optimizer / schedular
        # important: the following entries (optimizer_type and learning_rate) are only used
        # if the base model does not have a configure_optimizer method!
        optimizer_type: Optional[Union[str, Type[Optimizer]]] = None,
        learning_rate: Optional[float] = None,
        warmup_proportion: float = 0.0,
        scheduler_name: Optional[Union[str, SchedulerType]] = None,
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if base_model is None:
            if base_model_type is None:
                raise ValueError(
                    "Either base_model or base_model_type must be provided. If base_model is not provided, "
                    "base_model_type must be a valid model type, e.g. 'transformers.AutoModelForSeq2SeqLM'."
                )
            logger.warning(
                "The base_model_type and base_model_config arguments are deprecated. Please use base_model. "
                "You can use the following code to create the base_model argument: "
                "base_model = {'_type_': base_model_type, **base_model_config}"
            )
            base_model = {"_type_": base_model_type, **(base_model_config or {})}

        if scheduler_name is None and warmup_proportion > 0.0:
            logger.warning(
                "warmup_proportion is set to a value > 0.0, but scheduler_name is not set. "
                "Setting scheduler_name to 'linear' by default."
            )
            scheduler_name = "linear"

        self.save_hyperparameters(ignore=["base_model_type", "base_model_config"])

        # optimizer / scheduler
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.scheduler_name = scheduler_name
        self.warmup_proportion = warmup_proportion
        self.scheduler_kwargs = scheduler_kwargs or {}

        self.model = self.setup_base_model(config=base_model)
        self.generation_config = self.configure_generation(**(override_generation_kwargs or {}))

    def setup_base_model(self, config: Dict[str, Any]) -> PreTrainedModel:
        config = copy.copy(config)
        resolved_base_model_type: Type[PreTrainedModel] = resolve_type(config.pop("_type_"))
        return resolved_base_model_type.from_pretrained(**config)

    def configure_generation(self, **kwargs) -> Dict[str, Any]:
        if self.taskmodule is not None:
            # get the generation config from the taskmodule
            generation_config = self.taskmodule.configure_model_generation()
        else:
            logger.warning(
                "No taskmodule is available, so no generation config will be created. Consider "
                "setting taskmodule_config to a valid taskmodule config to use specific setup for generation."
            )
            generation_config = {}
        generation_config.update(kwargs)
        return generation_config

    def predict(self, inputs, **kwargs) -> TargetType:
        is_training = self.training
        self.eval()

        generation_kwargs = copy.deepcopy(self.generation_config)
        generation_kwargs.update(kwargs)
        outputs = self.model.generate(**inputs, **generation_kwargs)

        if is_training:
            self.train()

        # TODO: move into base model? or does this work for "all" generative models?
        # strip the bos_id
        if isinstance(outputs, torch.Tensor):
            labels = outputs[:, 1:]
            assert isinstance(labels, LongTensor)
            return {"labels": labels}
        else:
            raise ValueError(f"Unsupported output type: {type(outputs)}")

    def forward(self, inputs: InputType, targets: Optional[TargetType] = None) -> OutputType:
        kwargs = {**inputs, **(targets or {})}
        return self.model(**kwargs)

    def decode(self, inputs: InputType, outputs: OutputType) -> TargetType:
        # construct prediction from the model output
        logits = outputs.logits
        # get the indices (these are without the initial bos_ids, see above)
        prediction = torch.argmax(logits, dim=-1).to(torch.long)
        assert isinstance(prediction, LongTensor)
        return {"labels": prediction}

    def configure_optimizers(self) -> OptimizerLRScheduler:
        if hasattr(self.model, "configure_optimizer") and callable(self.model.configure_optimizer):
            if self.learning_rate is not None:
                raise ValueError(
                    f"learning_rate is set to {self.learning_rate}, but the *base model* ({type(self.model)}) has a "
                    f"configure_optimizer method. Please set learning_rate to None and configure the optimizer "
                    f"inside the *base model*."
                )
            optimizer = self.model.configure_optimizer()
        else:
            logger.warning(
                f"The model does not have a configure_optimizer method. Creating an optimizer of "
                f"optimizer_type={self.optimizer_type} with the learning_rate={self.learning_rate} instead."
            )
            if self.optimizer_type is None:
                raise ValueError(
                    f"optimizer_type is None, but the *base model* ({type(self.model)}) does not have a "
                    f"configure_optimizer method. Please set the optimizer_type to a valid optimizer type, "
                    f"e.g. optimizer_type=torch.optim.Adam."
                )
            resolved_optimizer_type = resolve_type(
                self.optimizer_type, expected_super_type=Optimizer
            )
            optimizer = resolved_optimizer_type(self.parameters(), lr=self.learning_rate)

        if self.scheduler_name is not None:
            num_training_steps = self.trainer.estimated_stepping_batches
            num_warmup_steps = int(num_training_steps * self.warmup_proportion)
            scheduler = get_scheduler(
                name=self.scheduler_name,
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                scheduler_specific_kwargs=self.scheduler_kwargs,
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        else:
            return optimizer
