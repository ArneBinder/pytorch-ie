import logging
from typing import Iterator, MutableMapping, Optional, Tuple, Union

import torch.nn
from pie_core import Model
from torch import FloatTensor, LongTensor
from torch.nn import Parameter
from torch.optim import AdamW
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from typing_extensions import TypeAlias

from pytorch_ie.models.common import ModelWithBoilerplate
from pytorch_ie.models.interface import RequiresModelNameOrPath, RequiresNumClasses

# model inputs / outputs / targets
InputType: TypeAlias = MutableMapping[str, LongTensor]
OutputType: TypeAlias = SequenceClassifierOutput
TargetType: TypeAlias = MutableMapping[str, Union[LongTensor, FloatTensor]]
# step inputs (batch) / outputs (loss)
StepInputType: TypeAlias = Tuple[InputType, TargetType]
StepOutputType: TypeAlias = FloatTensor


logger = logging.getLogger(__name__)


@Model.register()
class SimpleSequenceClassificationModel(
    ModelWithBoilerplate[InputType, OutputType, TargetType, StepOutputType],
    RequiresModelNameOrPath,
    RequiresNumClasses,
):
    """A simple sequence classification model. It wraps a HuggingFace
    AutoModelForSequenceClassification and adds boilerplate code for training and inference.

    Args:
        model_name_or_path: The name or path of the HuggingFace model to use.
        num_classes: The number of classes for the classification task.
        tokenizer_vocab_size: The size of the tokenizer vocabulary. If provided, the model's
            tokenizer embeddings are resized to this size.
        learning_rate: The learning rate for the optimizer.
        task_learning_rate: The learning rate for the task-specific parameters. If None, the
            learning rate for all parameters is set to `learning_rate`.
        warmup_proportion: The proportion of steps to warm up the learning rate.
        freeze_base_model: If True, the base model parameters are frozen.
        base_model_prefix: The prefix of the base model parameters when using a task_learning_rate
            or freeze_base_model. If None, the base_model_prefix of the model is used.
        **kwargs: Additional keyword arguments passed to the parent class,
            see :class:`ModelWithBoilerplate`.
    """

    def __init__(
        self,
        model_name_or_path: str,
        num_classes: int,
        tokenizer_vocab_size: Optional[int] = None,
        learning_rate: float = 1e-5,
        task_learning_rate: Optional[float] = None,
        warmup_proportion: float = 0.1,
        freeze_base_model: bool = False,
        base_model_prefix: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.task_learning_rate = task_learning_rate
        self.warmup_proportion = warmup_proportion
        self.freeze_base_model = freeze_base_model

        config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_classes)
        if self.is_from_pretrained:
            self.model = AutoModelForSequenceClassification.from_config(config=config)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path, config=config
            )

        self.base_model_prefix = base_model_prefix or self.model.base_model_prefix

        if tokenizer_vocab_size is not None:
            self.model.resize_token_embeddings(tokenizer_vocab_size)

        if self.freeze_base_model:
            for name, param in self.base_model_named_parameters():
                param.requires_grad = False

    def base_model_named_parameters(self, prefix: str = "") -> Iterator[Tuple[str, Parameter]]:
        base_model: Optional[torch.nn.Module] = getattr(self.model, self.base_model_prefix, None)
        if base_model is None:
            raise ValueError(
                f"Base model with prefix '{self.base_model_prefix}' not found in {type(self.model).__name__}"
            )
        if prefix:
            prefix = f"{prefix}."
        return base_model.named_parameters(prefix=f"{prefix}model.{self.base_model_prefix}")

    def task_named_parameters(self, prefix: str = "") -> Iterator[Tuple[str, Parameter]]:
        base_model_parameter_names = dict(self.base_model_named_parameters(prefix=prefix)).keys()
        for name, param in self.named_parameters(prefix=prefix):
            if name not in base_model_parameter_names:
                yield name, param

    def forward(self, inputs: InputType, targets: Optional[TargetType] = None) -> OutputType:
        kwargs = {**inputs, **(targets or {})}
        return self.model(**kwargs)

    def decode(self, inputs: InputType, outputs: OutputType) -> TargetType:
        labels = torch.argmax(outputs.logits, dim=-1).to(torch.long)
        probabilities = torch.softmax(outputs.logits, dim=-1)
        assert isinstance(labels, LongTensor)
        assert isinstance(probabilities, FloatTensor)
        return {"labels": labels, "probabilities": probabilities}

    def configure_optimizers(self):
        if self.task_learning_rate is not None:
            base_model_params = [param for name, param in self.base_model_named_parameters()]
            task_params = [param for name, param in self.task_named_parameters()]
            optimizer = AdamW(
                [
                    {"params": base_model_params, "lr": self.learning_rate},
                    {"params": task_params, "lr": self.task_learning_rate},
                ]
            )
        else:
            optimizer = AdamW(self.parameters(), lr=self.learning_rate)

        if self.warmup_proportion > 0.0:
            stepping_batches = self.trainer.estimated_stepping_batches
            scheduler = get_linear_schedule_with_warmup(
                optimizer, int(stepping_batches * self.warmup_proportion), stepping_batches
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        else:
            return optimizer
