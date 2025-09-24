import logging
from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    MutableMapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import torch
from pie_core import Model
from torch import FloatTensor, LongTensor, nn
from torch.nn import Parameter
from torch.optim import AdamW
from transformers import AutoConfig, AutoModel, PreTrainedModel, get_linear_schedule_with_warmup
from transformers.modeling_outputs import SequenceClassifierOutput
from typing_extensions import TypeAlias

from pytorch_ie.models.interface import RequiresModelNameOrPath, RequiresNumClasses

from .common import ModelWithBoilerplate
from .components.pooler import get_pooler_and_output_size

# model inputs / outputs / targets
InputType: TypeAlias = MutableMapping[str, LongTensor]
OutputType: TypeAlias = SequenceClassifierOutput
TargetType: TypeAlias = MutableMapping[str, Union[LongTensor, FloatTensor]]
# step inputs (batch) / outputs (loss)
StepInputType: TypeAlias = Tuple[InputType, TargetType]
StepOutputType: TypeAlias = FloatTensor


HF_MODEL_TYPE_TO_CLASSIFIER_DROPOUT_ATTRIBUTE = {
    "albert": "classifier_dropout_prob",
    "distilbert": "seq_classif_dropout",
}

logger = logging.getLogger(__name__)

T = TypeVar("T")


def separate_arguments_by_prefix(
    arguments: MutableMapping[str, T], prefixes: List[str]
) -> Dict[str, Dict[str, T]]:
    result: Dict[str, Dict[str, T]] = {prefix: {} for prefix in prefixes + ["remaining"]}
    for k, v in arguments.items():
        found = False
        for prefix in prefixes:
            if k.startswith(prefix):
                result[prefix][k[len(prefix) :]] = v
                found = True
                break
        if not found:
            result["remaining"][k] = v
    return result


class SequenceClassificationModelWithPoolerBase(
    ABC,
    ModelWithBoilerplate[InputType, OutputType, TargetType, StepOutputType],
    RequiresModelNameOrPath,
):
    """Abstract base model for sequence classification with a pooler.

    Args:
        model_name_or_path: The name or path of the HuggingFace model to use.
        tokenizer_vocab_size: The size of the tokenizer vocabulary. If provided, the model's
            tokenizer embeddings are resized to this size.
        classifier_dropout: The dropout probability for the classifier. If not provided, the
            dropout probability is taken from the Huggingface model config.
        learning_rate: The learning rate for the optimizer.
        task_learning_rate: The learning rate for the task-specific parameters. If None, the
            learning rate for all parameters is set to `learning_rate`.
        warmup_proportion: The proportion of steps to warm up the learning rate.
        pooler: The pooler configuration. If None, CLS token pooling is used.
        freeze_base_model: If True, the base model parameters are frozen.
        base_model_prefix: The prefix of the base model parameters when using a task_learning_rate
            or freeze_base_model. If None, the base_model_prefix of the model is used.
        **kwargs: Additional keyword arguments passed to the parent class,
            see :class:`ModelWithBoilerplate`.
    """

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer_vocab_size: Optional[int] = None,
        classifier_dropout: Optional[float] = None,
        learning_rate: float = 1e-5,
        task_learning_rate: Optional[float] = None,
        warmup_proportion: float = 0.1,
        pooler: Optional[Union[Dict[str, Any], str]] = None,
        freeze_base_model: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.task_learning_rate = task_learning_rate
        self.warmup_proportion = warmup_proportion
        self.freeze_base_model = freeze_base_model
        self.model_name_or_path = model_name_or_path

        self.model = self.setup_base_model()

        if tokenizer_vocab_size is not None:
            self.model.resize_token_embeddings(tokenizer_vocab_size)

        if self.freeze_base_model:
            for param in self.model.parameters():
                param.requires_grad = False

        if classifier_dropout is None:
            # Get the classifier dropout value from the Huggingface model config.
            # This is a bit of a mess since some Configs use different variable names or change the semantics
            # of the dropout (e.g. DistilBert has one dropout prob for QA and one for Seq classification, and a
            # general one for embeddings, encoder and pooler).
            classifier_dropout_attr = HF_MODEL_TYPE_TO_CLASSIFIER_DROPOUT_ATTRIBUTE.get(
                self.model.config.model_type, "classifier_dropout"
            )
            classifier_dropout = getattr(self.model.config, classifier_dropout_attr) or 0.0
        self.dropout = nn.Dropout(classifier_dropout)

        if isinstance(pooler, str):
            pooler = {"type": pooler}
        self.pooler_config = pooler or {}
        self.pooler, pooler_output_dim = self.setup_pooler(input_dim=self.model.config.hidden_size)
        self.classifier = self.setup_classifier(pooler_output_dim=pooler_output_dim)
        self.loss_fct = self.setup_loss_fct()

    def setup_base_model(self) -> PreTrainedModel:
        config = AutoConfig.from_pretrained(self.model_name_or_path)
        if self.is_from_pretrained:
            return AutoModel.from_config(config=config)
        else:
            return AutoModel.from_pretrained(self.model_name_or_path, config=config)

    @abstractmethod
    def setup_classifier(self, pooler_output_dim: int) -> Callable:
        pass

    @abstractmethod
    def setup_loss_fct(self) -> Callable:
        pass

    def setup_pooler(self, input_dim: int) -> Tuple[Callable, int]:
        """Set up the pooler. The pooler is used to get a representation of the input sequence(s)
        that can be used by the classifier. It is a callable that takes the hidden states of the
        base model (and additional model inputs that are prefixed with "pooler_") and returns the
        pooled output.

        Args:
            input_dim: The input dimension of the pooler, i.e. the hidden size of the base model.

        Returns:
            A tuple with the pooler and the output dimension of the pooler.
        """
        return get_pooler_and_output_size(config=self.pooler_config, input_dim=input_dim)

    def get_pooled_output(self, model_inputs, pooler_inputs) -> torch.FloatTensor:
        output = self.model(**model_inputs)
        hidden_state = output.last_hidden_state
        pooled_output = self.pooler(hidden_state, **pooler_inputs)
        pooled_output = self.dropout(pooled_output)
        return pooled_output

    def forward(
        self,
        inputs: InputType,
        targets: Optional[TargetType] = None,
        return_hidden_states: bool = False,
    ) -> OutputType:
        sanitized_inputs = separate_arguments_by_prefix(arguments=inputs, prefixes=["pooler_"])

        pooled_output = self.get_pooled_output(
            model_inputs=sanitized_inputs["remaining"], pooler_inputs=sanitized_inputs["pooler_"]
        )

        logits = self.classifier(pooled_output)

        result = {"logits": logits}
        if targets is not None:
            labels = targets["labels"]
            loss = self.loss_fct(logits, labels)
            result["loss"] = loss
        if return_hidden_states:
            raise NotImplementedError("return_hidden_states is not yet implemented")

        return SequenceClassifierOutput(**result)

    @abstractmethod
    def decode(self, inputs: InputType, outputs: OutputType) -> TargetType:
        pass

    def base_model_named_parameters(self, prefix: str = "") -> Iterator[Tuple[str, Parameter]]:
        if prefix:
            prefix = f"{prefix}."
        return self.model.named_parameters(prefix=f"{prefix}model")

    def task_named_parameters(self, prefix: str = "") -> Iterator[Tuple[str, Parameter]]:
        if prefix:
            prefix = f"{prefix}."
        base_model_parameter_names = dict(self.base_model_named_parameters(prefix=prefix)).keys()
        for name, param in self.named_parameters(prefix=prefix):
            if name not in base_model_parameter_names:
                yield name, param

    def configure_optimizers(self):
        if self.task_learning_rate is not None:
            base_model_params = (param for name, param in self.base_model_named_parameters())
            task_params = (param for name, param in self.task_named_parameters())
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


@Model.register()
class SequenceClassificationModelWithPooler(
    SequenceClassificationModelWithPoolerBase,
    RequiresNumClasses,
):
    """A sequence classification model that uses a pooler to get a representation of the input
    sequence and then applies a linear classifier to that representation. The pooler can be
    configured via the `pooler` argument, see :func:`get_pooler_and_output_size` for details.

    Args:
        num_classes: The number of classes for the classification task.
        multi_label: If True, the model is trained as a multi-label classifier.
        multi_label_threshold: The threshold for the multi-label classifier, i.e. the probability
            above which a class is predicted.
        **kwargs
    """

    def __init__(
        self,
        num_classes: int,
        multi_label: bool = False,
        multi_label_threshold: float = 0.5,
        **kwargs,
    ):
        # set num_classes and multi_label before call to super init because they are used there
        # in setup_classifier and setup_loss_fct
        self.num_classes = num_classes
        self.multi_label = multi_label
        super().__init__(**kwargs)

        self.multi_label_threshold = multi_label_threshold

    def setup_classifier(self, pooler_output_dim: int) -> Callable:
        return nn.Linear(pooler_output_dim, self.num_classes)

    def setup_loss_fct(self) -> Callable:
        return nn.BCEWithLogitsLoss() if self.multi_label else nn.CrossEntropyLoss()

    def decode(self, inputs: InputType, outputs: OutputType) -> TargetType:
        if not self.multi_label:
            labels = torch.argmax(outputs.logits, dim=-1).to(torch.long)
            probabilities = torch.softmax(outputs.logits, dim=-1)
        else:
            probabilities = torch.sigmoid(outputs.logits)
            labels = (probabilities > self.multi_label_threshold).to(torch.long)
        assert isinstance(probabilities, FloatTensor)
        assert isinstance(labels, LongTensor)
        return {"labels": labels, "probabilities": probabilities}


@Model.register()
class SequencePairSimilarityModelWithPooler(
    SequenceClassificationModelWithPoolerBase,
):
    """A span pair similarity model to detect of two spans occurring in different texts are
    similar. It uses an encoder to independently calculate contextualized embeddings of both texts,
    then uses a pooler to get representations of the spans and, finally, calculates the cosine to
    get the similarity scores.

    Args:
        label_threshold: The threshold above which score the spans are considered as similar.
        pooler: The pooler identifier or config, see :func:`get_pooler_and_output_size` for details.
            Defaults to "mention_pooling" (max pooling over the span token embeddings).
        **kwargs
    """

    def __init__(
        self,
        pooler: Optional[Union[Dict[str, Any], str]] = None,
        **kwargs,
    ):
        if pooler is None:
            # use (max) mention pooling per default
            pooler = {"type": "mention_pooling", "num_indices": 1}
        super().__init__(pooler=pooler, **kwargs)

    def setup_classifier(
        self, pooler_output_dim: int
    ) -> Callable[[torch.FloatTensor, torch.FloatTensor], torch.FloatTensor]:
        return torch.nn.functional.cosine_similarity  # type: ignore[return-value]

    def setup_loss_fct(self) -> Callable:
        return nn.BCELoss()

    def forward(
        self,
        inputs: InputType,
        targets: Optional[TargetType] = None,
        return_hidden_states: bool = False,
    ) -> OutputType:
        sanitized_inputs = separate_arguments_by_prefix(
            # Note that the order of the prefixes is important because one is a prefix of the other,
            # so we need to start with the longer!
            arguments=inputs,
            prefixes=["pooler_pair_", "pooler_"],
        )

        pooled_output = self.get_pooled_output(
            model_inputs=sanitized_inputs["remaining"]["encoding"],
            pooler_inputs=sanitized_inputs["pooler_"],
        )
        pooled_output_pair = self.get_pooled_output(
            model_inputs=sanitized_inputs["remaining"]["encoding_pair"],
            pooler_inputs=sanitized_inputs["pooler_pair_"],
        )

        logits = self.classifier(pooled_output, pooled_output_pair)

        result = {"logits": logits}
        if targets is not None:
            labels = targets["scores"]
            loss = self.loss_fct(logits, labels)
            result["loss"] = loss
        if return_hidden_states:
            raise NotImplementedError("return_hidden_states is not yet implemented")

        return SequenceClassifierOutput(**result)

    def decode(self, inputs: InputType, outputs: OutputType) -> TargetType:
        # probabilities = torch.sigmoid(outputs.logits)
        scores = outputs.logits
        return {"scores": scores}
