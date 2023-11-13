"""
workflow:
    document
        -> (input_encoding, target_encoding) -> task_encoding
            -> model_encoding -> model_output
        -> task_output
    -> document
"""

import logging
import re
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Type, Union

from transformers import AutoTokenizer
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy
from typing_extensions import TypeAlias

from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.core import Annotation, TaskEncoding, TaskModule
from pytorch_ie.documents import TextDocument, TextDocumentWithLabeledSpansAndBinaryRelations
from pytorch_ie.models.transformer_seq2seq import ModelOutputType, ModelStepInputType

InputEncodingType: TypeAlias = Dict[str, Sequence[int]]
TargetEncodingType: TypeAlias = Dict[str, Sequence[int]]

TaskEncodingType: TypeAlias = TaskEncoding[TextDocument, InputEncodingType, TargetEncodingType]
TaskOutputType: TypeAlias = Sequence[Dict[str, Any]]

TaskModuleType: TypeAlias = TaskModule[
    TextDocument,
    InputEncodingType,
    TargetEncodingType,
    ModelStepInputType,
    ModelOutputType,
    TaskOutputType,
]

logger = logging.getLogger(__name__)


@TaskModule.register()
class TransformerSeq2SeqTaskModule(TaskModuleType):

    DOCUMENT_TYPE = TextDocumentWithLabeledSpansAndBinaryRelations

    def __init__(
        self,
        tokenizer_name_or_path: str,
        entity_annotation: str = "labeled_spans",
        relation_annotation: str = "binary_relations",
        padding: Union[bool, str, PaddingStrategy] = True,
        truncation: Union[bool, str, TruncationStrategy] = True,
        max_input_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.relation_annotation = relation_annotation
        self.entity_annotation = entity_annotation
        self.padding = padding
        self.truncation = truncation
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.pad_to_multiple_of = pad_to_multiple_of

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    @property
    def document_type(self) -> Optional[Type[TextDocument]]:
        dt: Type[TextDocument] = self.DOCUMENT_TYPE
        if (
            self.entity_annotation == "labeled_spans"
            and self.relation_annotation == "binary_relations"
        ):
            return dt
        else:
            logger.warning(
                f"entity_annotation={self.entity_annotation} and relation_annotation={self.relation_annotation} are "
                f"not the default values ('labeled_spans' and 'binary_relations'), so the taskmodule "
                f"{type(self).__name__} can not request the usual document type ({dt.__name__}) for auto-conversion "
                f"because this has the bespoken default values as layer names instead of the provided ones."
            )
            return None

    def encode_text(self, text: str) -> InputEncodingType:
        return self.tokenizer(
            text,
            padding=False,
            truncation=self.truncation,
            max_length=self.max_input_length,
            is_split_into_words=False,
        )

    def encode_input(
        self,
        document: TextDocument,
    ) -> Optional[Union[TaskEncodingType, Sequence[TaskEncodingType],]]:
        return TaskEncoding(
            document=document,
            inputs=self.encode_text(document.text),
        )

    def document_to_target_string(self, document: TextDocument) -> str:
        relations: Sequence[BinaryRelation] = document[self.relation_annotation]

        head_to_tail_and_label: Dict[LabeledSpan, List[Tuple[LabeledSpan, str]]] = {}
        for relation in relations:
            if not isinstance(relation.head, LabeledSpan) or not isinstance(
                relation.tail, LabeledSpan
            ):
                raise TypeError(
                    f"the taskmodule expects the relation arguments to be of type LabeledSpan, "
                    f"but got {type(relation.head)} and {type(relation.tail)}"
                )
            if relation.head not in head_to_tail_and_label:
                head_to_tail_and_label[relation.head] = []

            head_to_tail_and_label[relation.head].append((relation.tail, relation.label))

        lin_triplets: List[str] = []
        for head in sorted(head_to_tail_and_label.keys(), key=lambda head: head.start):
            tail_and_label_for_head = head_to_tail_and_label[head]

            head_entity = document.text[head.start : head.end]

            lin_triplets.append("<triplet>")
            lin_triplets.append(head_entity)

            for tail, label in sorted(
                tail_and_label_for_head, key=lambda tail_and_label: tail_and_label[0].start
            ):
                tail_entity = document.text[tail.start : tail.end]

                lin_triplets.append("<subj>")
                lin_triplets.append(tail_entity)
                lin_triplets.append("<obj>")
                lin_triplets.append(label)

        return " ".join(lin_triplets)

    def encode_target(
        self,
        task_encoding: TaskEncodingType,
    ) -> TargetEncodingType:
        target_string = self.document_to_target_string(task_encoding.document)
        return {"labels": self.encode_text(target_string)["input_ids"]}

    def unbatch_output(self, model_output: ModelOutputType) -> Sequence[TaskOutputType]:
        unbatched_output = []
        for out in model_output:
            decoded_string = self.tokenizer.decode(
                out, skip_special_tokens=False, clean_up_tokenization_spaces=True
            )
            triplets = self._extract_triplets(decoded_string)
            unbatched_output.append(triplets)

        return unbatched_output

    def create_annotations_from_output(
        self,
        task_encoding: TaskEncodingType,
        task_output: TaskOutputType,
    ) -> Iterator[Tuple[str, Annotation]]:
        for relation_dct in task_output:
            head_entity = relation_dct["head"]
            tail_entity = relation_dct["tail"]
            label = relation_dct["type"]

            if label == "no_relation":
                continue

            # for now, just use the first head and tail match in the document
            text = task_encoding.document.text.lower()
            try:
                # this may fail if head_entity or tail_entity contains any special regex character (e.g. brackets)
                head_match = re.search(head_entity.lower(), text)
                tail_match = re.search(tail_entity.lower(), text)
            except Exception:
                logger.warning(
                    f"could not successfully search for the entities in the text, skip the triplet "
                    f'(head: "{head_entity}", tail: "{tail_entity}", label: "{label}")'
                )
                continue

            if head_match is None or tail_match is None:
                continue

            head = LabeledSpan(start=head_match.start(), end=head_match.end(), label="head")
            tail = LabeledSpan(start=tail_match.start(), end=tail_match.end(), label="tail")

            relation = BinaryRelation(head=head, tail=tail, label=label)

            yield from [
                (self.entity_annotation, head),
                (self.entity_annotation, tail),
                (self.relation_annotation, relation),
            ]

    def collate(self, task_encodings: Sequence[TaskEncodingType]) -> ModelStepInputType:
        input_features = [task_encoding.inputs for task_encoding in task_encodings]

        padded_encoding = self.tokenizer.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_input_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if task_encodings[0].has_targets:
            # TODO: this is a bit of a hack -- fix
            labels = {
                "input_ids": [task_encoding.targets["labels"] for task_encoding in task_encodings]
            }

            padded_labels = self.tokenizer.pad(
                labels,
                padding=self.padding,
                max_length=self.max_target_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )

            # TODO: this is a bit of a hack -- fix
            padded_encoding["labels"] = padded_labels["input_ids"]

        return (padded_encoding,)

    # TODO: improve this method as soon as we have unittests for this taskmodule
    def _extract_triplets(self, text) -> TaskOutputType:
        triplets = []
        relation, subject, relation, object_ = "", "", "", ""
        text = text.strip()
        current = "x"
        for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
            if token == "<triplet>":
                current = "t"
                if relation != "":
                    triplets.append(
                        {
                            "head": subject.strip(),
                            "type": relation.strip(),
                            "tail": object_.strip(),
                        }
                    )
                    relation = ""
                subject = ""
            elif token == "<subj>":
                current = "s"
                if relation != "":
                    triplets.append(
                        {
                            "head": subject.strip(),
                            "type": relation.strip(),
                            "tail": object_.strip(),
                        }
                    )
                object_ = ""
            elif token == "<obj>":
                current = "o"
                relation = ""
            else:
                if current == "t":
                    subject += " " + token
                elif current == "s":
                    object_ += " " + token
                elif current == "o":
                    relation += " " + token
        if subject != "" and relation != "" and object_ != "":
            triplets.append(
                {"head": subject.strip(), "type": relation.strip(), "tail": object_.strip()}
            )
        return triplets
