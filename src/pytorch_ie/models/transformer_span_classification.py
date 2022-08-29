from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torchmetrics
from torch import Tensor, nn
from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    BatchEncoding,
    get_linear_schedule_with_warmup,
)

from pytorch_ie.core import PyTorchIEModel
from pytorch_ie.models.modules.mlp import MLP

TransformerSpanClassificationModelBatchEncoding = BatchEncoding
TransformerSpanClassificationModelBatchOutput = Dict[str, Any]

TransformerSpanClassificationModelStepBatchEncoding = Tuple[
    Dict[str, Tensor],
    Optional[Sequence[Sequence[Tuple[int, int, int]]]],
]


TRAINING = "train"
VALIDATION = "val"
TEST = "test"


@PyTorchIEModel.register()
class TransformerSpanClassificationModel(PyTorchIEModel):
    def __init__(
        self,
        model_name_or_path: str,
        num_classes: int,
        t_total: int,
        learning_rate: float = 1e-5,
        task_learning_rate: float = 1e-4,
        warmup_proportion: float = 0.1,
        ignore_index: int = 0,
        max_span_length: int = 8,
        span_length_embedding_dim: int = 150,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.t_total = t_total
        self.learning_rate = learning_rate
        self.task_learning_rate = task_learning_rate
        self.warmup_proportion = warmup_proportion
        self.max_span_length = max_span_length

        config = AutoConfig.from_pretrained(model_name_or_path)
        if self.is_from_pretrained:
            self.model = AutoModel.from_config(config=config)
        else:
            self.model = AutoModel.from_pretrained(model_name_or_path, config=config)

        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        # TODO: properly intialize!
        # self.classifier = nn.Linear(config.hidden_size * 2 + span_length_embedding_dim, num_classes)
        self.classifier = MLP(
            input_dim=config.hidden_size * 2 + span_length_embedding_dim,
            output_dim=num_classes,
            hidden_dim=150,
            num_layers=2,
        )

        self.span_length_embedding = nn.Embedding(
            num_embeddings=max_span_length, embedding_dim=span_length_embedding_dim
        )

        self.loss_fct = nn.CrossEntropyLoss()

        self.f1 = nn.ModuleDict(
            {
                f"stage_{stage}": torchmetrics.F1Score(
                    num_classes=num_classes, ignore_index=ignore_index
                )
                for stage in [TRAINING, VALIDATION, TEST]
            }
        )

    def _start_end_and_span_length_span_index(
        self, batch_size: int, max_seq_length: int, seq_lengths: Optional[Iterable[int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if seq_lengths is None:
            seq_lengths = batch_size * [max_seq_length]

        start_indices = []
        end_indices = []
        span_lengths = []
        span_batch_index = []
        offsets = []
        for batch_index, seq_length in enumerate(seq_lengths):
            offset = max_seq_length * batch_index

            for span_length in range(1, self.max_span_length + 1):
                for start_index in range(seq_length + 1 - span_length):
                    end_index = start_index + span_length - 1

                    span_batch_index.append(batch_index)
                    start_indices.append(start_index)
                    end_indices.append(end_index)
                    span_lengths.append(span_length - 1)
                    offsets.append(offset)

        return (
            torch.tensor(start_indices),
            torch.tensor(end_indices),
            torch.tensor(span_lengths),
            torch.tensor(span_batch_index),
            torch.tensor(offsets),
        )

    # TODO: this should live in the taskmodule
    def _expand_target_tuples(
        self,
        target_tuples: Sequence[Sequence[Tuple[int, int, int]]],
        batch_size: int,
        max_seq_length: int,
        seq_lengths: Optional[Iterable[int]] = None,
    ) -> torch.Tensor:
        if seq_lengths is None:
            seq_lengths = batch_size * [max_seq_length]

        target = []
        for batch_index, seq_length in enumerate(seq_lengths):
            label_lookup = {
                (start, end): label for start, end, label in target_tuples[batch_index]
            }
            for span_length in range(1, self.max_span_length + 1):
                for start_index in range(seq_length + 1 - span_length):
                    end_index = start_index + span_length - 1

                    label = label_lookup.get((start_index, end_index), 0)
                    target.append(label)

        return torch.tensor(target)

    def forward(self, input_: TransformerSpanClassificationModelBatchEncoding) -> TransformerSpanClassificationModelBatchOutput:  # type: ignore
        output = self.model(**input_)

        batch_size, seq_length, hidden_dim = output.last_hidden_state.shape
        hidden_state = output.last_hidden_state.view(batch_size * seq_length, hidden_dim)

        seq_lengths = None
        if "attention_mask" in input_:
            seq_lengths = torch.sum(input_["attention_mask"], dim=-1).detach().cpu()

        (
            start_indices,
            end_indices,
            span_length,
            batch_indices,
            offsets,
        ) = self._start_end_and_span_length_span_index(
            batch_size=batch_size, max_seq_length=seq_length, seq_lengths=seq_lengths
        )

        start_embedding = hidden_state[offsets + start_indices, :]
        end_embedding = hidden_state[offsets + end_indices, :]
        span_length_embedding = self.span_length_embedding(span_length.to(hidden_state.device))

        combined_embedding = torch.cat(
            (start_embedding, end_embedding, span_length_embedding), dim=-1
        )

        logits = self.classifier(self.dropout(combined_embedding))

        return {
            "logits": logits,
            "batch_indices": batch_indices,
            "start_indices": start_indices,
            "end_indices": end_indices,
        }

    def step(self, stage: str, batch: TransformerSpanClassificationModelStepBatchEncoding, batch_idx):  # type: ignore
        input_, target_tuples = batch
        assert target_tuples is not None, f"target has to be available for {stage}"

        output = self(input_)

        logits = output["logits"]

        batch_size, seq_length = input_["input_ids"].shape
        seq_lengths = None
        if "attention_mask" in input_:
            seq_lengths = torch.sum(input_["attention_mask"], dim=-1)

        # TODO: Why is this not happening in TransformerSpanClassificationTaskModule.collate?
        target = self._expand_target_tuples(
            target_tuples=target_tuples,
            batch_size=batch_size,
            max_seq_length=seq_length,
            seq_lengths=seq_lengths,
        )
        target = target.to(logits.device)

        loss = self.loss_fct(logits, target)

        self.log(f"{stage}/loss", loss, on_step=stage == TRAINING, on_epoch=True, prog_bar=True)

        f1 = self.f1[f"stage_{stage}"]
        f1(logits, target)
        self.log(f"{stage}/f1", f1, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def training_step(self, batch: TransformerSpanClassificationModelStepBatchEncoding, batch_idx: int):  # type: ignore
        return self.step(stage=TRAINING, batch=batch, batch_idx=batch_idx)

    def validation_step(self, batch: TransformerSpanClassificationModelStepBatchEncoding, batch_idx: int):  # type: ignore
        return self.step(stage=VALIDATION, batch=batch, batch_idx=batch_idx)

    def test_step(self, batch: TransformerSpanClassificationModelStepBatchEncoding, batch_idx: int):  # type: ignore
        return self.step(stage=TEST, batch=batch, batch_idx=batch_idx)

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        # TODO: this needs fixing (does not work models other than BERT)
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if "bert" in n]},
            {
                "params": [p for n, p in param_optimizer if "bert" not in n],
                "lr": self.task_learning_rate,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(self.t_total * self.warmup_proportion), self.t_total
        )
        return [optimizer], [scheduler]
        # return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
