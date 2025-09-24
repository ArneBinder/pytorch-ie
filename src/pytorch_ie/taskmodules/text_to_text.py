import dataclasses
import logging
from functools import partial
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Type, Union

import torch
from pie_core import Annotation, AnnotationLayer, Document, TaskEncoding, TaskModule
from pie_core.taskmodule import InputEncoding, ModelBatchOutput, TargetEncoding, TaskBatchEncoding
from pie_core.utils.hydra import resolve_type
from pie_documents.annotations import AnnotationWithText
from pie_documents.document.processing import token_based_document_to_text_based
from pie_documents.documents import TextBasedDocument, TokenBasedDocument
from torchmetrics import Metric
from transformers import AutoTokenizer, PreTrainedTokenizer
from typing_extensions import TypeAlias

from ..utils.document import tokenize_document
from .common import BatchableMixin, get_first_occurrence_index
from .metrics import WrappedMetricWithPrepareFunction

logger = logging.getLogger(__name__)


DocumentType: TypeAlias = TextBasedDocument


@dataclasses.dataclass
class InputEncodingType(BatchableMixin):
    input_ids: List[int]
    attention_mask: List[int]


@dataclasses.dataclass
class TargetEncodingType(BatchableMixin):
    labels: List[int]
    # this is optional because we use the same type for TaskOutputType, which does not have this field
    decoder_attention_mask: Optional[List[int]] = None


TaskEncodingType: TypeAlias = TaskEncoding[
    DocumentType,
    InputEncodingType,
    TargetEncodingType,
]
TaskOutputType: TypeAlias = TargetEncodingType


# we use a custom un-batch function for metrics, because the text metrics such as ROUGEScore metric expects
# strings for input and target
def unbatch_and_untokenize(
    batch: ModelBatchOutput, taskmodule: "TextToTextTaskModule"
) -> Sequence[str]:
    unbatched = taskmodule.unbatch_output(batch)
    texts = [
        taskmodule.tokenizer.decode(encoding.labels, skip_special_tokens=True)
        for encoding in unbatched
    ]
    return texts


@TaskModule.register()
class TextToTextTaskModule(
    TaskModule[
        DocumentType,
        InputEncoding,
        TargetEncoding,
        TaskBatchEncoding,
        ModelBatchOutput,
        TaskOutputType,
    ],
):
    """A PIE task module for text-to-text tasks. It works with simple text annotations, e.g.
    abstractive summaries, as target annotations.

    It can also be used with additional guidance annotations, e.g. questions for generative question answering, in
    which case the text of the guidance annotation is prepended to the input text.

    Args:
        tokenizer_name_or_path: The name (Huggingface Hub model identifier) or local path of the tokenizer to use.
        document_type: The type of the input document. Must be a string that resolves to a subclass of
            TextBasedDocument, e.g. "pie_documents.documents.TextDocumentWithAbstractiveSummary" for abstractive
            summarization.
        tokenized_document_type: The type of the tokenized document. Must be a string that resolves to a
            subclass of TokenBasedDocument, e.g. "pie_documents.documents.TokenDocumentWithAbstractiveSummary" for
            abstractive summarization.
        target_layer: The name of the annotation layer that contains the target annotations, e.g. "abstractive_summary"
            for abstractive summarization.
        target_annotation_type: The type of the target annotations. Must be a string that resolves to a subclass
            of AnnotationWithText, e.g. "pie_documents.annotations.AbstractiveSummary" for abstractive summarization.
        guidance_layer: The name of the annotation layer that contains the guidance annotations. If set, the text of
            the guidance annotation is prepended to the input text.
        guidance_annotation_field: The name of the field in the target annotations that contains the guidance
            annotation. Required if guidance_layer is defined to attach the guidance annotation to the newly created
            target annotation.
        text_metric_type: The type of the text metric to use for evaluation. Must be a string that resolves to a
            subclass of Metric, e.g. "torchmetrics.text.ROUGEScore" for ROUGE score.
        tokenizer_init_kwargs: Additional keyword arguments that are passed to the tokenizer constructor.
        tokenizer_kwargs: Additional keyword arguments that are passed when calling the tokenizer.
        partition_layer_name: The name of the annotation layer that contains the partitions. If set, the partitions
            will be used to split the input text into multiple parts which are then tokenized separately. This can be
            used to split long documents into multiple parts to avoid exceeding the maximum input length of the
            tokenizer / model.
        annotation_field_mapping: A mapping from input document annotation layer names to layer names defined in the
            document_type / tokenized_document_type. This can be used if the actual input documents have different
            annotation layer names than the provided document_type / tokenized_document_type.
        log_first_n_examples: The number of examples to log. If set to a positive integer n, the first n examples will
            be logged. This can be used to check if the input and target encodings are as expected.
    """

    def __init__(
        self,
        tokenizer_name_or_path: str,
        document_type: str,
        tokenized_document_type: str,
        target_layer: str,
        target_annotation_type: str,
        guidance_layer: Optional[str] = None,
        guidance_annotation_field: Optional[str] = None,
        text_metric_type: Optional[str] = None,
        tokenizer_init_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        partition_layer_name: Optional[str] = None,
        annotation_field_mapping: Optional[Dict[str, str]] = None,
        log_first_n_examples: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.target_layer = target_layer
        self.guidance_layer = guidance_layer
        self.target_annotation_type: Type[AnnotationWithText] = resolve_type(
            target_annotation_type, expected_super_type=AnnotationWithText
        )
        self.guidance_annotation_field = guidance_annotation_field
        self.text_metric_type: Optional[Metric] = None
        if text_metric_type is not None:
            self.text_metric_type = resolve_type(text_metric_type, expected_super_type=Metric)

        # tokenization
        self._document_type: Type[TextBasedDocument] = resolve_type(
            document_type, expected_super_type=TextBasedDocument
        )
        self._tokenized_document_type: Type[TokenBasedDocument] = resolve_type(
            tokenized_document_type, expected_super_type=TokenBasedDocument
        )
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            **(tokenizer_init_kwargs or {}),
        )
        self.annotation_field_mapping = annotation_field_mapping or dict()
        self.partition_layer_name = partition_layer_name

        # target encoding
        self.pad_values = {
            "input_ids": self.tokenizer.pad_token_id,
            "attention_mask": 0,
            "labels": self.tokenizer.pad_token_id,
            "decoder_attention_mask": 0,
        }
        self.dtypes = {
            "input_ids": torch.int64,
            "attention_mask": torch.int64,
            "labels": torch.int64,
            "decoder_attention_mask": torch.int64,
        }

        # logging
        self.log_first_n_examples = log_first_n_examples

    @property
    def document_type(self) -> Type[TextBasedDocument]:
        return self._document_type

    @property
    def tokenized_document_type(self) -> Type[TokenBasedDocument]:
        return self._tokenized_document_type

    @property
    def layer_names(self) -> List[str]:
        return [self.target_layer]

    def get_mapped_layer(self, document: Document, layer_name: str) -> AnnotationLayer:
        if layer_name in self.annotation_field_mapping:
            layer_name = self.annotation_field_mapping[layer_name]
        return document[layer_name]

    @property
    def generation_config(self) -> Dict[str, Any]:
        return {}

    def maybe_log_example(
        self,
        task_encoding: TaskEncodingType,
        targets: Optional[TargetEncodingType] = None,
    ) -> None:
        if self.log_first_n_examples is not None and self.log_first_n_examples > 0:
            inputs = task_encoding.inputs

            logger.info(f"input_ids: {inputs.input_ids}")
            logger.info(f"attention_mask: {inputs.attention_mask}")
            if targets is not None or task_encoding.has_targets:
                targets = targets or task_encoding.targets
                logger.info(f"labels: {targets.labels}")
            self.log_first_n_examples -= 1

    def warn_only_once(self, message: str) -> None:
        if not hasattr(self, "_warned"):
            self._warned: Set[str] = set()
        if message not in self._warned:
            logger.warning(f"{message} (This warning will only be shown once)")
            self._warned.add(message)

    def encode_annotations(
        self,
        layers: Dict[str, AnnotationLayer],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TargetEncodingType:
        target_annotations = []
        guidance_annotation = (
            metadata.get("guidance_annotation", None) if metadata is not None else None
        )
        if guidance_annotation is not None:
            if self.guidance_annotation_field is None:
                raise ValueError(
                    "guidance_annotation is available, but guidance_annotation_field is not set"
                )
            # filter annotations that belong to the guidance_annotation
            for target_annotation in layers[self.target_layer]:
                current_guidance_annotation = getattr(
                    target_annotation, self.guidance_annotation_field
                )
                if current_guidance_annotation == guidance_annotation:
                    target_annotations.append(target_annotation)
        else:
            target_annotations = layers[self.target_layer]

        if len(target_annotations) == 0:
            raise ValueError(f"target_annotations {self.target_layer} contains no annotation")
        elif len(target_annotations) > 1:
            self.warn_only_once(
                f"target_annotations {self.target_layer} contains more than one annotation, "
                f"but only the first one will be used"
            )
        annotation = target_annotations[0]
        if isinstance(annotation, self.target_annotation_type):
            text = target_annotations[0].text
        else:
            raise ValueError(
                f"target_annotations {self.target_layer} contains an annotation of type {type(annotation)}, "
                f"but expected {self.target_annotation_type}"
            )
        encoding = self.tokenizer(text)
        return TargetEncodingType(
            labels=encoding["input_ids"], decoder_attention_mask=encoding["attention_mask"]
        )

    def decode_annotations(
        self, encoding: TaskOutputType, metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, List[Annotation]], Any]:
        text = self.tokenizer.decode(encoding.labels, skip_special_tokens=True)
        annotation_kwargs = {}
        if self.guidance_annotation_field is not None:
            if metadata is None:
                raise ValueError(
                    "metadata is required to decode annotations with guidance_annotation_field"
                )
            guidance_annotation = metadata.get("guidance_annotation", None)
            if guidance_annotation is not None:
                if self.guidance_annotation_field is None:
                    raise ValueError(
                        "guidance_annotation is available, but guidance_annotation_field is not set"
                    )
                annotation_kwargs[self.guidance_annotation_field] = guidance_annotation

        decoded_layers = {
            self.target_layer: [self.target_annotation_type(text=text, **annotation_kwargs)]
        }
        # no error collection yet
        errors: Dict[str, Any] = {}
        return decoded_layers, errors

    def tokenize_document(
        self, document: DocumentType, source_text: Optional[str] = None
    ) -> List[TokenBasedDocument]:
        field_mapping = dict(self.annotation_field_mapping)
        if self.partition_layer_name is not None:
            field_mapping[self.partition_layer_name] = "labeled_partitions"
            partition_layer = "labeled_partitions"
        else:
            partition_layer = None
        casted_document = document.as_type(self.document_type, field_mapping=field_mapping)

        tokenizer_kwargs = dict(self.tokenizer_kwargs)
        if source_text is not None:
            tokenizer_kwargs["text"] = source_text
        tokenized_docs = tokenize_document(
            casted_document,
            tokenizer=self.tokenizer,
            result_document_type=self.tokenized_document_type,
            partition_layer=partition_layer,
            **tokenizer_kwargs,
        )
        for idx, tokenized_doc in enumerate(tokenized_docs):
            tokenized_doc.id = f"{document.id}-tokenized-{idx+1}-of-{len(tokenized_docs)}"

        return tokenized_docs

    def encode_input(
        self, document: DocumentType, is_training: bool = False
    ) -> Optional[Union[TaskEncodingType, Sequence[TaskEncodingType]]]:
        task_encodings: List[TaskEncodingType] = []
        if self.guidance_layer is None:
            guidance_annotations = [None]
        else:
            guidance_annotations = document[self.guidance_layer]
        for guidance_annotation in guidance_annotations:
            source_text = None
            if guidance_annotation is not None:
                # Here could also more sophisticated logic be implemented
                source_text = guidance_annotation.text
            tokenized_docs = self.tokenize_document(document, source_text=source_text)
            for tokenized_doc in tokenized_docs:
                tokenizer_encoding = tokenized_doc.metadata["tokenizer_encoding"]
                task_encodings.append(
                    TaskEncoding(
                        document=document,
                        inputs=InputEncodingType(
                            input_ids=tokenizer_encoding.ids,
                            attention_mask=tokenizer_encoding.attention_mask,
                        ),
                        metadata={
                            "tokenized_document": tokenized_doc,
                            "guidance_annotation": guidance_annotation,
                        },
                    )
                )

        return task_encodings

    def encode_target(self, task_encoding: TaskEncodingType) -> Optional[TargetEncodingType]:
        document = task_encoding.metadata["tokenized_document"]
        guidance_annotation = task_encoding.metadata["guidance_annotation"]

        layers = {
            layer_name: self.get_mapped_layer(document, layer_name=layer_name)
            for layer_name in self.layer_names
        }
        result = self.encode_annotations(
            layers=layers,
            metadata={**task_encoding.metadata, "guidance_annotation": guidance_annotation},
        )

        self.maybe_log_example(task_encoding=task_encoding, targets=result)
        return result

    def collate(self, task_encodings: Sequence[TaskEncodingType]) -> TaskBatchEncoding:
        if len(task_encodings) == 0:
            raise ValueError("no task_encodings available")
        inputs = InputEncodingType.batch(
            values=[x.inputs for x in task_encodings],
            dtypes=self.dtypes,
            pad_values=self.pad_values,
        )

        targets = None
        if task_encodings[0].has_targets:
            targets = TargetEncodingType.batch(
                values=[x.targets for x in task_encodings],
                dtypes=self.dtypes,
                pad_values=self.pad_values,
            )

        return inputs, targets

    def unbatch_output(self, model_output: ModelBatchOutput) -> Sequence[TaskOutputType]:
        labels = model_output["labels"]
        batch_size = labels.size(0)

        # We use the position after the first eos token as the seq_len.
        # Note that, if eos_id is not in model_output for a given batch item, the result will be
        # model_output.size(1) + 1 (i.e. seq_len + 1) for that batch item. This is fine, because we use the
        # seq_lengths just to truncate the output and want to keep everything if eos_id is not present.
        seq_lengths = get_first_occurrence_index(labels, self.tokenizer.eos_token_id) + 1

        result = [
            TaskOutputType(labels[i, : seq_lengths[i]].to(device="cpu").tolist())
            for i in range(batch_size)
        ]
        return result

    def create_annotations_from_output(
        self,
        task_encoding: TaskEncodingType,
        task_output: TaskOutputType,
    ) -> Iterator[Tuple[str, Annotation]]:
        layers, errors = self.decode_annotations(
            encoding=task_output, metadata=task_encoding.metadata
        )
        tokenized_document = task_encoding.metadata["tokenized_document"]

        # Note: token_based_document_to_text_based() does not yet consider predictions, so we need to clear
        # the main annotations and attach the predictions to that
        for layer_name, annotations in layers.items():
            layer = self.get_mapped_layer(tokenized_document, layer_name=layer_name)
            layer.clear()
            layer.extend(annotations)

        untokenized_document = token_based_document_to_text_based(
            tokenized_document, result_document_type=self.document_type
        )

        for layer_name in layers:
            annotations = self.get_mapped_layer(untokenized_document, layer_name=layer_name)
            for annotation in annotations:
                yield layer_name, annotation.copy()

    def configure_model_generation(self) -> Optional[Dict[str, Any]]:
        # we do not set any overrides here, because we want to use the default generation config as
        # it is derived from the Huggingface base model config.json
        return {}

    def configure_model_metric(self, stage: str) -> Optional[Metric]:
        if self.text_metric_type is None:
            return None

        return WrappedMetricWithPrepareFunction(
            metric=self.text_metric_type(),
            prepare_function=partial(unbatch_and_untokenize, taskmodule=self),
            prepare_does_unbatch=True,
        )
