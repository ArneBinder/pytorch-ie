import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy

from pytorch_ie.data.document import Document, LabeledSpan
from pytorch_ie.taskmodules.taskmodule import TaskEncoding, TaskModule


class TransformerSetPredictionTaskModule(TaskModule):
    def __init__(
        self,
        tokenizer_name_or_path: str,
        entity_annotation: str = "entities",
        single_sentence: bool = False,
        sentence_annotation: str = "sentences",
        padding: Union[bool, str, PaddingStrategy] = True,
        truncation: Union[bool, str, TruncationStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        label_pad_token_id: int = -100,
    ) -> None:
        super().__init__(
            tokenizer_name_or_path=tokenizer_name_or_path,
            entity_annotation=entity_annotation,
            single_sentence=single_sentence,
            sentence_annotation=sentence_annotation,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            label_pad_token_id=label_pad_token_id,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.entity_annotation = entity_annotation
        self.single_sentence = single_sentence
        self.sentence_annotation = sentence_annotation
        self.label_to_id = {}
        self.id_to_label = {}
        self.padding = padding
        self.truncation = truncation
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id

    def prepare(self, documents: List[Document]) -> None:
        labels = set()
        for document in documents:
            entities = document.annotations(self.entity_annotation)

            for entity in entities:
                entity_labels = entity.label if entity.is_multilabel else [entity.label]
                for label in entity_labels:
                    if label not in labels:
                        labels.add(label)

        # self.label_to_id["O"] = 0
        # current_id = 1
        current_id = 0
        for label in labels:
            self.label_to_id[label] = current_id
            current_id += 1

        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

    def encode_input(
        self, documents: List[Document]
    ) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]], Optional[List[Document]]]:
        expanded_documents = None
        if self.single_sentence:
            input_ = [
                self.tokenizer(
                    doc.text[sent.start : sent.end],
                    padding=False,
                    truncation=self.truncation,
                    max_length=self.max_length,
                    is_split_into_words=False,
                    return_offsets_mapping=True,
                    return_special_tokens_mask=True,
                )
                for doc in documents
                for sent in doc.annotations(self.sentence_annotation)
            ]
            expanded_documents = [
                doc for doc in documents for _ in doc.annotations(self.sentence_annotation)
            ]
        else:
            input_ = [
                self.tokenizer(
                    doc.text,
                    padding=False,
                    truncation=self.truncation,
                    max_length=self.max_length,
                    is_split_into_words=False,
                    return_offsets_mapping=True,
                    return_special_tokens_mask=True,
                )
                for doc in documents
            ]

        metadata = [
            {
                "offset_mapping": inp.pop("offset_mapping"),
                "special_tokens_mask": inp.pop("special_tokens_mask"),
            }
            for inp in input_
        ]

        if self.single_sentence:
            i = 0
            for document in documents:
                for sentence_index in range(len(document.annotations(self.sentence_annotation))):
                    metadata[i]["sentence_index"] = sentence_index
                    i += 1

        return input_, metadata, expanded_documents

    def encode_target(
        self, documents: List[Document], input_: List[Dict[str, Any]]
    ) -> Union[List[List[int]], List[Dict[str, Any]]]:
        target = []
        if self.single_sentence:
            i = 0
            for document in documents:
                entities = document.annotations(self.entity_annotation)
                sentences = document.annotations(self.sentence_annotation)

                for sentence in sentences:
                    start_indices = []
                    end_indices = []
                    label_ids = []
                    span_masks = []
                    for entity in entities:
                        if entity.start < sentence.start or entity.end > sentence.end:
                            continue

                        entity_start = entity.start - sentence.start
                        entity_end = entity.end - sentence.start

                        start_idx = input_[i].char_to_token(entity_start)
                        end_idx = input_[i].char_to_token(entity_end - 1)

                        if start_idx is None or end_idx is None:
                            continue

                        start_indices.append(start_idx)
                        end_indices.append(end_idx)
                        label_ids.append(self.label_to_id[entity.label])

                        span_mask = [
                            1 if start_idx <= i <= end_idx else 0
                            for i in range(len(input_[i].word_ids()))
                        ]
                        span_masks.append(span_mask)

                    target.append(
                        {
                            "entities": {
                                "start_index": start_indices,
                                "end_index": end_indices,
                                "label_ids": label_ids,
                                # "span_position": span_positions,
                                "span_mask": span_masks,
                            }
                        }
                    )
                    i += 1
        else:
            for i, document in enumerate(documents):
                entities = document.annotations(self.entity_annotation)

                start_indices = []
                end_indices = []
                label_ids = []
                span_masks = []
                for entity in entities:
                    start_idx = input_[i].char_to_token(entity.start)
                    end_idx = input_[i].char_to_token(entity.end - 1)

                    if start_idx is None or end_idx is None:
                        continue

                    start_indices.append(start_idx)
                    end_indices.append(end_idx)
                    label_ids.append(self.label_to_id[entity.label])

                    span_mask = [
                        1 if start_idx <= i <= end_idx else 0
                        for i in range(len(input_[i].word_ids()))
                    ]
                    span_masks.append(span_mask)

                target.append(
                    {
                        "entities": {
                            "start_index": start_indices,
                            "end_index": end_indices,
                            "label_ids": label_ids,
                            # "span_position": span_positions,
                            "span_mask": span_masks,
                        }
                    }
                )

        return target

    def decode_output(self, output: Dict[str, Any]) -> List[Dict[str, Any]]:
        predictions = output["entities"]

        label_ids_pred_full = (
            F.softmax(predictions["label_ids"], dim=-1).argmax(dim=-1).cpu().numpy()
        )
        start_index_pred_full = (
            F.softmax(predictions["start_index"], dim=-1).argmax(dim=-1).cpu().numpy()
        )
        end_index_pred_full = (
            F.softmax(predictions["end_index"], dim=-1).argmax(dim=-1).cpu().numpy()
        )

        tags = []
        probabilities = []
        for label_ids_pred, start_index_pred, end_index_pred in zip(
            label_ids_pred_full, start_index_pred_full, end_index_pred_full
        ):
            indices_pred = label_ids_pred != len(self.label_to_id)

            label_ids = label_ids_pred[indices_pred]
            start_indices = start_index_pred[indices_pred]
            end_indices = end_index_pred[indices_pred]

            current_tags = []
            current_probabilities = []
            for label_id, start, end in zip(label_ids, start_indices, end_indices):
                label = self.id_to_label[label_id]
                current_tags.append((label, (start, end)))
                current_probabilities.append(1.0)

            tags.append(current_tags)
            probabilities.append(current_probabilities)

        # labels = [[self.id_to_label[e] for e in b] for b in label_ids]
        return [{"tags": t, "probabilities": p} for t, p in zip(tags, probabilities)]

    def combine(
        self,
        encodings: List[TaskEncoding],
        outputs: List[Dict[str, Any]],
    ) -> None:
        if self.single_sentence:
            for encoding, output in zip(encodings, outputs):
                document = encoding.document
                metadata = encoding.metadata

                sentence = document.annotations(self.sentence_annotation)[
                    metadata["sentence_index"]
                ]

                # tag_sequence = [
                #     "O" if stm else tag
                #     for tag, stm in zip(output["tags"], metadata["special_tokens_mask"])
                # ]

                # spans = bio_tags_to_spans(tag_sequence)
                spans = output["tags"]
                for label, (start, end) in spans:
                    document.add_prediction(
                        self.entity_annotation,
                        LabeledSpan(
                            sentence.start + metadata["offset_mapping"][start][0],
                            sentence.start + metadata["offset_mapping"][end][1],
                            label,
                        ),
                    )
        else:
            for encoding, output in zip(encodings, outputs):
                document = encoding.document
                metadata = encoding.metadata

                # tag_sequence = [
                #     "O" if stm else tag
                #     for tag, stm in zip(output["tags"], metadata["special_tokens_mask"])
                # ]

                # spans = bio_tags_to_spans(tag_sequence)
                spans = output["tags"]
                for label, (start, end) in spans:
                    document.add_prediction(
                        self.entity_annotation,
                        LabeledSpan(
                            metadata["offset_mapping"][start][0],
                            metadata["offset_mapping"][end][1],
                            label,
                        ),
                    )

    def collate(self, encodings: List[TaskEncoding]) -> Dict[str, Any]:
        input_features = [encoding.input for encoding in encodings]
        metadata = [encoding.metadata for encoding in encodings]
        documents = [encoding.document for encoding in encodings]

        input_ = self.tokenizer.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if encodings[0].target is None:
            return input_, None, metadata, documents

        target: Dict[str, Dict[str, List[torch.Tensor]]] = {}
        for encoding in encodings:
            for decoder, decoder_target in encoding.target.items():
                if decoder not in target:
                    target[decoder] = {}
                for key, val in decoder_target.items():
                    if key not in target[decoder]:
                        target[decoder][key] = []
                    if val and isinstance(val[0], list):
                        if key.endswith("_mask"):
                            seq_length = input_["input_ids"].shape[1]
                            val_tensor = []
                            for v in val:
                                padding_length = seq_length - len(v)
                                val_tensor.append(v + padding_length * [-100])

                            target[decoder][key].append(
                                torch.tensor(val_tensor, dtype=torch.float)
                            )
                        else:
                            target[decoder][key].append(
                                [torch.tensor(v, dtype=torch.float) for v in val]
                            )
                    else:
                        target[decoder][key].append(torch.tensor(val, dtype=torch.float))

        input_ = {k: torch.tensor(v, dtype=torch.int64) for k, v in input_.items()}

        return input_, target, metadata, documents

    @classmethod
    def from_pretrained(cls, save_directory: str) -> "TaskModule":
        task_module_config_file = os.path.join(save_directory, "task_module.json")

        with open(task_module_config_file, "r", encoding="utf-8") as f:
            init_kwargs = json.load(f)

        task_module_class = init_kwargs.pop("task_module_class")
        label_to_id = init_kwargs.pop("label_to_id")

        task_module = cls(**init_kwargs)

        if label_to_id is not None:
            task_module.label_to_id = label_to_id
            task_module.id_to_label = {v: k for k, v in label_to_id.items()}

        return task_module

    def save_pretrained(self, save_directory: str) -> None:
        if os.path.isfile(save_directory):
            return

        os.makedirs(save_directory, exist_ok=True)

        task_module_config_file = os.path.join(save_directory, "task_module.json")

        task_module_config = self.init_kwargs
        task_module_config["task_module_class"] = self.__class__.__name__
        task_module_config["label_to_id"] = self.label_to_id

        with open(task_module_config_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(task_module_config, ensure_ascii=False))
