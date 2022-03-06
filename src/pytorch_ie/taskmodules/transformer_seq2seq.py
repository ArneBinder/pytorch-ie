import logging
import re
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from transformers import AutoTokenizer
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy

from pytorch_ie.data.document import Annotation, BinaryRelation, Document, LabeledSpan
from pytorch_ie.models import (
    TransformerSeq2SeqModelBatchOutput,
    TransformerSeq2SeqModelStepBatchEncoding,
)
from pytorch_ie.taskmodules.taskmodule import Metadata, TaskEncoding, TaskModule

"""
workflow:
    Document
        -> (InputEncoding, TargetEncoding) -> TaskEncoding -> TaskBatchEncoding
            -> ModelBatchEncoding -> ModelBatchOutput
        -> TaskOutput
    -> Document
"""

TransformerSeq2SeqInputEncoding = Dict[str, List[int]]
TransformerSeq2SeqTargetEncoding = Dict[str, List[int]]

TransformerSeq2SeqTaskEncoding = TaskEncoding[
    TransformerSeq2SeqInputEncoding, TransformerSeq2SeqTargetEncoding
]
TransformerSeq2SeqTaskOutput = List[Dict[str, Any]]

_TransformerSeq2SeqTaskModule = TaskModule[
    # _InputEncoding, _TargetEncoding, _TaskBatchEncoding, _ModelBatchOutput, _TaskOutput
    TransformerSeq2SeqInputEncoding,
    TransformerSeq2SeqTargetEncoding,
    TransformerSeq2SeqModelStepBatchEncoding,
    TransformerSeq2SeqModelBatchOutput,
    TransformerSeq2SeqTaskOutput,
]

logger = logging.getLogger(__name__)


class TransformerSeq2SeqTaskModule(_TransformerSeq2SeqTaskModule):
    def __init__(
        self,
        tokenizer_name_or_path: str,
        relation_annotation: str = "relations",
        padding: Union[bool, str, PaddingStrategy] = True,
        truncation: Union[bool, str, TruncationStrategy] = True,
        max_input_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.relation_annotation = relation_annotation
        self.padding = padding
        self.truncation = truncation
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.pad_to_multiple_of = pad_to_multiple_of

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    def document_to_input_string(self, document: Document) -> str:
        return document.text

    def encode_input_strings(self, inputs: List[str]) -> List[TransformerSeq2SeqInputEncoding]:
        return [
            self.tokenizer(
                input_,
                padding=False,
                truncation=self.truncation,
                max_length=self.max_input_length,
                is_split_into_words=False,
            )
            for input_ in inputs
        ]

    def encode_input(
        self,
        documents: List[Document],
        is_training: bool = False,
    ) -> Tuple[List[TransformerSeq2SeqInputEncoding], List[Metadata], Optional[List[Document]]]:
        input_strings = [self.document_to_input_string(document) for document in documents]
        return (
            self.encode_input_strings(input_strings),
            [{} for _ in range(len(documents))],
            documents,
        )

    def document_to_target_string(self, document: Document) -> str:
        relations = document.annotations.binary_relations[self.relation_annotation]

        head_to_relation: Dict[LabeledSpan, List[BinaryRelation]] = {}
        for relation in relations:
            if relation.head not in head_to_relation:
                head_to_relation[relation.head] = []

            head_to_relation[relation.head].append(relation)

        all_relation_heads = {relation.head for relation in relations}

        lin_triplets: List[str] = []
        for head in sorted(all_relation_heads, key=lambda head: head.start):
            relations_with_head = head_to_relation[head]

            head_entity = document.text[head.start : head.end]

            lin_triplets.append("<triplet>")
            lin_triplets.append(head_entity)

            for tail_relation in sorted(
                relations_with_head, key=lambda relation: relation.tail.start
            ):
                tail_entity = document.text[tail_relation.tail.start : tail_relation.tail.end]

                if tail_relation.is_multilabel:
                    raise NotImplementedError

                label = tail_relation.label_single

                lin_triplets.append("<subj>")
                lin_triplets.append(tail_entity)
                lin_triplets.append("<obj>")
                lin_triplets.append(label)

        return " ".join(lin_triplets)

    def encode_target_strings(self, targets: List[str]) -> List[TransformerSeq2SeqTargetEncoding]:
        return [
            {
                "labels": self.tokenizer(
                    target,
                    padding=False,
                    truncation=self.truncation,
                    max_length=self.max_target_length,
                    is_split_into_words=False,
                )["input_ids"]
            }
            for target in targets
        ]

    def encode_target(
        self,
        documents: List[Document],
        input_encodings: List[TransformerSeq2SeqInputEncoding],
        metadata: List[Metadata],
    ) -> List[TransformerSeq2SeqTargetEncoding]:
        target_strings = [self.document_to_target_string(document) for document in documents]
        return self.encode_target_strings(target_strings)

    def unbatch_output(
        self, output: TransformerSeq2SeqModelBatchOutput
    ) -> Sequence[TransformerSeq2SeqTaskOutput]:
        unbatched_output = []
        for out in output:
            decoded_string = self.tokenizer.decode(
                out, skip_special_tokens=False, clean_up_tokenization_spaces=True
            )
            triplets = self._extract_triplets(decoded_string)
            unbatched_output.append(triplets)

        return unbatched_output

    def create_annotations_from_output(
        self,
        encoding: TransformerSeq2SeqTaskEncoding,
        output: TransformerSeq2SeqTaskOutput,
    ) -> Iterator[Tuple[str, Annotation]]:
        for relation in output:
            head_entity = relation["head"]
            tail_entity = relation["tail"]
            label = relation["type"]

            if label == "no_relation":
                continue

            # for now, just use the first head and tail match in the document
            text = encoding.document.text.lower()
            head_match = re.search(head_entity.lower(), text)
            tail_match = re.search(tail_entity.lower(), text)

            if head_match is None or tail_match is None:
                continue

            head = LabeledSpan(start=head_match.start(), end=head_match.end(), label="head")
            tail = LabeledSpan(start=tail_match.start(), end=tail_match.end(), label="tail")

            yield (
                self.relation_annotation,
                BinaryRelation(
                    head=head,
                    tail=tail,
                    label=label,
                ),
            )

    def collate(
        self, encodings: List[TransformerSeq2SeqTaskEncoding]
    ) -> TransformerSeq2SeqModelStepBatchEncoding:
        input_features = [encoding.input for encoding in encodings]
        metadata = [encoding.metadata for encoding in encodings]
        documents = [encoding.document for encoding in encodings]

        padded_encoding = self.tokenizer.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_input_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if encodings[0].has_target:
            # TODO: this is a bit of a hack -- fix
            labels = {"input_ids": [encoding.target["labels"] for encoding in encodings]}

            padded_labels = self.tokenizer.pad(
                labels,
                padding=self.padding,
                max_length=self.max_target_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )

            # TODO: this is a bit of a hack -- fix
            padded_encoding["labels"] = padded_labels["input_ids"]

        return padded_encoding, None, metadata, documents

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
