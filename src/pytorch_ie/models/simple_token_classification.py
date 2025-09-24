import logging
from typing import MutableMapping, Optional, Tuple, Union

import torch
from pie_core import Model
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import FloatTensor, LongTensor
from transformers import AutoConfig, AutoModelForTokenClassification, BatchEncoding
from transformers.modeling_outputs import TokenClassifierOutput
from typing_extensions import TypeAlias

from pytorch_ie.models.interface import RequiresModelNameOrPath, RequiresNumClasses

from .common import ModelWithBoilerplate

# model inputs / outputs / targets
InputType: TypeAlias = BatchEncoding
OutputType: TypeAlias = TokenClassifierOutput
TargetType: TypeAlias = MutableMapping[str, Union[LongTensor, FloatTensor]]
# step inputs (batch) / outputs (loss)
StepInputType: TypeAlias = Tuple[InputType, TargetType]
StepOutputType: TypeAlias = FloatTensor


logger = logging.getLogger(__name__)


@Model.register()
class SimpleTokenClassificationModel(
    ModelWithBoilerplate[InputType, OutputType, TargetType, StepOutputType],
    RequiresModelNameOrPath,
    RequiresNumClasses,
):
    """A simple token classification model that wraps a (pretrained) model loaded with
    AutoModelForTokenClassification from the transformers library.

    The model is trained with a cross-entropy loss function and uses the Adam optimizer.

    Note that for training, the labels for the special tokens (as well as for padding tokens)
    are expected to have the value label_pad_id (-100 by default, which is the default ignore_index
    value for the CrossEntropyLoss). The predictions for these tokens are also replaced with
    label_pad_id to match the training labels for correct metric calculation. Therefore, the model
    requires the special_tokens_mask and attention_mask (for padding) to be passed as inputs.

    Args:
        model_name_or_path: The name or path of the pretrained transformer model to use.
        num_classes: The number of classes to predict.
        learning_rate: The learning rate to use for training.
        label_pad_id: The label id to use for padding labels (at the padding token positions
            as well as for the special tokens).
    """

    def __init__(
        self,
        model_name_or_path: str,
        num_classes: int,
        learning_rate: float = 1e-5,
        label_pad_id: int = -100,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.label_pad_id = label_pad_id
        self.num_classes = num_classes

        config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_classes)
        if self.is_from_pretrained:
            self.model = AutoModelForTokenClassification.from_config(config=config)
        else:
            self.model = AutoModelForTokenClassification.from_pretrained(
                model_name_or_path, config=config
            )

    def forward(self, inputs: InputType, targets: Optional[TargetType] = None) -> OutputType:
        inputs_without_special_tokens_mask = {
            k: v for k, v in inputs.items() if k != "special_tokens_mask"
        }
        return self.model(**inputs_without_special_tokens_mask, **(targets or {}))

    def decode(self, inputs: InputType, outputs: OutputType) -> TargetType:
        # get the max index for each token from the logits
        tags_tensor = torch.argmax(outputs.logits, dim=-1)

        # mask out the padding and special tokens
        tags_tensor = tags_tensor.masked_fill(inputs["attention_mask"] == 0, self.label_pad_id)

        # mask out the special tokens
        tags_tensor = tags_tensor.masked_fill(
            inputs["special_tokens_mask"] == 1, self.label_pad_id
        )
        labels = tags_tensor.to(torch.long)
        probabilities = torch.softmax(outputs.logits, dim=-1)
        assert isinstance(labels, LongTensor)
        assert isinstance(probabilities, FloatTensor)
        return {"labels": labels, "probabilities": probabilities}

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
