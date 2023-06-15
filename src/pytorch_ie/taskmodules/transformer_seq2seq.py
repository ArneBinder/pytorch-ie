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
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from transformers import AutoTokenizer
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy
from typing_extensions import TypeAlias

from pytorch_ie.annotations import BinaryRelation, LabeledSpan, Span
from pytorch_ie.core import Annotation, TaskEncoding, TaskModule
from pytorch_ie.documents import TextDocument
from pytorch_ie.models import (
    TransformerSeq2SeqModelBatchOutput,
    TransformerSeq2SeqModelStepBatchEncoding,
)

TransformerSeq2SeqInputEncoding: TypeAlias = Dict[str, Sequence[int]]
TransformerSeq2SeqTargetEncoding: TypeAlias = Dict[str, Sequence[int]]

TransformerSeq2SeqTaskEncoding: TypeAlias = TaskEncoding[
    TextDocument, TransformerSeq2SeqInputEncoding, TransformerSeq2SeqTargetEncoding
]
TransformerSeq2SeqTaskOutput: TypeAlias = Sequence[Dict[str, Any]]

_TransformerSeq2SeqTaskModule: TypeAlias = TaskModule[
    # _InputEncoding, _TargetEncoding, _TaskBatchEncoding, _ModelBatchOutput, _TaskOutput
    TextDocument,
    TransformerSeq2SeqInputEncoding,
    TransformerSeq2SeqTargetEncoding,
    TransformerSeq2SeqModelStepBatchEncoding,
    TransformerSeq2SeqModelBatchOutput,
    TransformerSeq2SeqTaskOutput,
]

logger = logging.getLogger(__name__)


@TaskModule.register()
class TransformerSeq2SeqTaskModule(_TransformerSeq2SeqTaskModule):
    def __init__(
        self,
        tokenizer_name_or_path: str,
        entity_annotation: str = "entities",
        relation_annotation: str = "relations",
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

    def encode_text(self, text: str) -> TransformerSeq2SeqInputEncoding:
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
        is_training: bool = False,
    ) -> Optional[
        Union[
            TransformerSeq2SeqTaskEncoding,
            Sequence[TransformerSeq2SeqTaskEncoding],
        ]
    ]:
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
        task_encoding: TransformerSeq2SeqTaskEncoding,
    ) -> TransformerSeq2SeqTargetEncoding:
        target_string = self.document_to_target_string(task_encoding.document)
        return {"labels": self.encode_text(target_string)["input_ids"]}

    def unbatch_output(
        self, model_output: TransformerSeq2SeqModelBatchOutput
    ) -> Sequence[TransformerSeq2SeqTaskOutput]:
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
        task_encoding: TransformerSeq2SeqTaskEncoding,
        task_output: TransformerSeq2SeqTaskOutput,
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

    def collate(
        self, task_encodings: Sequence[TransformerSeq2SeqTaskEncoding]
    ) -> TransformerSeq2SeqModelStepBatchEncoding:
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
    def _extract_triplets(self, text) -> TransformerSeq2SeqTaskOutput:
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
