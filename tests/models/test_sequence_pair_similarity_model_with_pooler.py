from typing import Dict

import pytest
import torch
from pytorch_lightning import Trainer
from torch import FloatTensor, LongTensor, tensor
from torch.optim.lr_scheduler import LambdaLR
from transformers.modeling_outputs import SequenceClassifierOutput

from pytorch_ie.models import SequencePairSimilarityModelWithPooler
from pytorch_ie.models.sequence_classification_with_pooler import OutputType
from tests.models import trunc_number


@pytest.fixture
def inputs() -> Dict[str, LongTensor]:
    result_dict = {
        "encoding": {
            "input_ids": tensor(
                [
                    [101, 1262, 1131, 1771, 140, 119, 102],
                    [101, 1262, 1131, 1771, 140, 119, 102],
                    [101, 1262, 1131, 1771, 140, 119, 102],
                    [101, 1262, 1131, 1771, 140, 119, 102],
                ]
            ),
            "token_type_ids": tensor(
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ]
            ),
            "attention_mask": tensor(
                [
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                ]
            ),
        },
        "encoding_pair": {
            "input_ids": tensor(
                [
                    [101, 3162, 7871, 1117, 5855, 119, 102],
                    [101, 3162, 7871, 1117, 5855, 119, 102],
                    [101, 3162, 7871, 1117, 5855, 119, 102],
                    [101, 3162, 7871, 1117, 5855, 119, 102],
                ]
            ),
            "token_type_ids": tensor(
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ]
            ),
            "attention_mask": tensor(
                [
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                ]
            ),
        },
        "pooler_start_indices": tensor([[2], [2], [4], [4]]),
        "pooler_end_indices": tensor([[3], [3], [5], [5]]),
        "pooler_pair_start_indices": tensor([[1], [3], [1], [3]]),
        "pooler_pair_end_indices": tensor([[2], [5], [2], [5]]),
    }

    return result_dict  # type: ignore[return-value]


@pytest.fixture
def targets() -> Dict[str, FloatTensor]:
    scores: FloatTensor = tensor([0.0, 0.0, 0.0, 0.0])  # type: ignore[assignment]
    return {"scores": scores}


@pytest.fixture
def model() -> SequencePairSimilarityModelWithPooler:
    torch.manual_seed(42)
    result = SequencePairSimilarityModelWithPooler(
        model_name_or_path="prajjwal1/bert-tiny",
    )
    return result


def test_model(model):
    assert model is not None
    named_parameters = dict(model.named_parameters())
    parameter_means = {k: trunc_number(v.mean().item(), 7) for k, v in named_parameters.items()}
    parameter_means_expected = {
        "model.embeddings.word_embeddings.weight": 0.0031152,
        "model.embeddings.position_embeddings.weight": 5.5e-05,
        "model.embeddings.token_type_embeddings.weight": -0.0015419,
        "model.embeddings.LayerNorm.weight": 1.312345,
        "model.embeddings.LayerNorm.bias": -0.0294608,
        "model.encoder.layer.0.attention.self.query.weight": -0.0003949,
        "model.encoder.layer.0.attention.self.query.bias": 0.0185744,
        "model.encoder.layer.0.attention.self.key.weight": 0.0003863,
        "model.encoder.layer.0.attention.self.key.bias": 0.0020557,
        "model.encoder.layer.0.attention.self.value.weight": 4.22e-05,
        "model.encoder.layer.0.attention.self.value.bias": 0.0065417,
        "model.encoder.layer.0.attention.output.dense.weight": 3.01e-05,
        "model.encoder.layer.0.attention.output.dense.bias": 0.0007209,
        "model.encoder.layer.0.attention.output.LayerNorm.weight": 1.199831,
        "model.encoder.layer.0.attention.output.LayerNorm.bias": 0.0608714,
        "model.encoder.layer.0.intermediate.dense.weight": -0.0011731,
        "model.encoder.layer.0.intermediate.dense.bias": -0.1219958,
        "model.encoder.layer.0.output.dense.weight": -0.0002212,
        "model.encoder.layer.0.output.dense.bias": -0.0013031,
        "model.encoder.layer.0.output.LayerNorm.weight": 1.2419648,
        "model.encoder.layer.0.output.LayerNorm.bias": 0.005295,
        "model.encoder.layer.1.attention.self.query.weight": -0.0007321,
        "model.encoder.layer.1.attention.self.query.bias": -0.0358397,
        "model.encoder.layer.1.attention.self.key.weight": 0.0001333,
        "model.encoder.layer.1.attention.self.key.bias": 0.0045062,
        "model.encoder.layer.1.attention.self.value.weight": 0.0001012,
        "model.encoder.layer.1.attention.self.value.bias": -0.0007094,
        "model.encoder.layer.1.attention.output.dense.weight": -2.43e-05,
        "model.encoder.layer.1.attention.output.dense.bias": 0.0041446,
        "model.encoder.layer.1.attention.output.LayerNorm.weight": 1.0377343,
        "model.encoder.layer.1.attention.output.LayerNorm.bias": 0.0443237,
        "model.encoder.layer.1.intermediate.dense.weight": -0.001344,
        "model.encoder.layer.1.intermediate.dense.bias": -0.1247257,
        "model.encoder.layer.1.output.dense.weight": -5.32e-05,
        "model.encoder.layer.1.output.dense.bias": 0.000677,
        "model.encoder.layer.1.output.LayerNorm.weight": 1.017162,
        "model.encoder.layer.1.output.LayerNorm.bias": -0.0474442,
        "model.pooler.dense.weight": 0.0001295,
        "model.pooler.dense.bias": -0.0052078,
        "pooler.missing_embeddings": 0.0812017,
    }
    assert parameter_means == parameter_means_expected


def test_model_pickleable(model):
    import pickle

    pickle.dumps(model)


@pytest.fixture
def model_output(model, inputs) -> OutputType:
    # set seed to make sure the output is deterministic
    torch.manual_seed(42)
    return model(inputs)


def test_forward_logits(model_output, inputs):
    assert isinstance(model_output, SequenceClassifierOutput)

    logits = model_output.logits

    torch.testing.assert_close(
        logits,
        torch.tensor(
            [0.5338148474693298, 0.5866107940673828, 0.5076886415481567, 0.5946245789527893]
        ),
    )


def test_decode(model, model_output, inputs):
    decoded = model.decode(inputs=inputs, outputs=model_output)
    assert isinstance(decoded, dict)
    assert set(decoded) == {"scores"}
    scores = decoded["scores"]
    torch.testing.assert_close(
        scores,
        torch.tensor(
            [0.5338148474693298, 0.5866107940673828, 0.5076886415481567, 0.5946245789527893]
        ),
    )


@pytest.fixture
def batch(inputs, targets):
    return inputs, targets


def test_training_step(batch, model):
    # set the seed to make sure the loss is deterministic
    torch.manual_seed(42)
    loss = model.training_step(batch, batch_idx=0)
    assert loss is not None
    torch.testing.assert_close(loss, torch.tensor(0.8145309686660767))


def test_validation_step(batch, model):
    # set the seed to make sure the loss is deterministic
    torch.manual_seed(42)
    loss = model.validation_step(batch, batch_idx=0)
    assert loss is not None
    torch.testing.assert_close(loss, torch.tensor(0.8145309686660767))


def test_test_step(batch, model):
    # set the seed to make sure the loss is deterministic
    torch.manual_seed(42)
    loss = model.test_step(batch, batch_idx=0)
    assert loss is not None
    torch.testing.assert_close(loss, torch.tensor(0.8145309686660767))


def test_base_model_named_parameters(model):
    base_model_named_parameters = dict(model.base_model_named_parameters())
    assert set(base_model_named_parameters) == {
        "model.pooler.dense.bias",
        "model.encoder.layer.0.intermediate.dense.weight",
        "model.encoder.layer.0.intermediate.dense.bias",
        "model.encoder.layer.1.attention.output.dense.weight",
        "model.encoder.layer.1.attention.output.LayerNorm.weight",
        "model.encoder.layer.1.attention.self.query.weight",
        "model.encoder.layer.1.output.dense.weight",
        "model.encoder.layer.0.output.dense.bias",
        "model.encoder.layer.1.intermediate.dense.bias",
        "model.encoder.layer.1.attention.self.value.bias",
        "model.encoder.layer.0.attention.output.dense.weight",
        "model.encoder.layer.0.attention.self.query.bias",
        "model.encoder.layer.0.attention.self.value.bias",
        "model.encoder.layer.1.output.dense.bias",
        "model.encoder.layer.1.attention.self.query.bias",
        "model.encoder.layer.1.attention.output.LayerNorm.bias",
        "model.encoder.layer.0.attention.self.query.weight",
        "model.encoder.layer.0.attention.output.LayerNorm.bias",
        "model.encoder.layer.0.attention.self.key.bias",
        "model.encoder.layer.1.intermediate.dense.weight",
        "model.encoder.layer.1.output.LayerNorm.bias",
        "model.encoder.layer.1.output.LayerNorm.weight",
        "model.encoder.layer.0.attention.self.key.weight",
        "model.encoder.layer.1.attention.output.dense.bias",
        "model.encoder.layer.0.attention.output.dense.bias",
        "model.embeddings.LayerNorm.bias",
        "model.encoder.layer.0.attention.self.value.weight",
        "model.encoder.layer.0.attention.output.LayerNorm.weight",
        "model.embeddings.token_type_embeddings.weight",
        "model.encoder.layer.0.output.LayerNorm.weight",
        "model.embeddings.position_embeddings.weight",
        "model.encoder.layer.1.attention.self.key.bias",
        "model.embeddings.LayerNorm.weight",
        "model.encoder.layer.0.output.LayerNorm.bias",
        "model.encoder.layer.1.attention.self.key.weight",
        "model.pooler.dense.weight",
        "model.encoder.layer.0.output.dense.weight",
        "model.embeddings.word_embeddings.weight",
        "model.encoder.layer.1.attention.self.value.weight",
    }


def test_task_named_parameters(model):
    task_named_parameters = dict(model.task_named_parameters())
    assert set(task_named_parameters) == {
        "pooler.missing_embeddings",
    }


def test_configure_optimizers_with_warmup():
    model = SequencePairSimilarityModelWithPooler(
        model_name_or_path="prajjwal1/bert-tiny",
    )
    model.trainer = Trainer(max_epochs=10)
    optimizers_and_schedulers = model.configure_optimizers()
    assert len(optimizers_and_schedulers) == 2
    optimizers, schedulers = optimizers_and_schedulers
    assert len(optimizers) == 1
    assert len(schedulers) == 1
    optimizer = optimizers[0]
    assert optimizer is not None
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["lr"] == 1e-05
    assert optimizer.defaults["weight_decay"] == 0.01
    assert optimizer.defaults["eps"] == 1e-08

    scheduler = schedulers[0]
    assert isinstance(scheduler, dict)
    assert set(scheduler) == {"scheduler", "interval"}
    assert isinstance(scheduler["scheduler"], LambdaLR)


def test_configure_optimizers_with_task_learning_rate(monkeypatch):
    model = SequencePairSimilarityModelWithPooler(
        model_name_or_path="prajjwal1/bert-tiny",
        learning_rate=1e-5,
        task_learning_rate=1e-3,
        # disable warmup to make sure the scheduler is not added which would set the learning rate
        # to 0
        warmup_proportion=0.0,
    )
    optimizer = model.configure_optimizers()
    assert optimizer is not None
    assert isinstance(optimizer, torch.optim.AdamW)
    assert len(optimizer.param_groups) == 2
    # base model parameters
    param_group = optimizer.param_groups[0]
    assert len(param_group["params"]) == 39
    assert param_group["lr"] == 1e-5
    # classifier head parameters - there is just the default embedding (which is not used)
    param_group = optimizer.param_groups[1]
    assert len(param_group["params"]) == 1
    assert param_group["lr"] == 1e-3
    # ensure that all parameters are covered
    assert set(optimizer.param_groups[0]["params"] + optimizer.param_groups[1]["params"]) == set(
        model.parameters()
    )


def test_freeze_base_model(monkeypatch, inputs, targets):
    model = SequencePairSimilarityModelWithPooler(
        model_name_or_path="prajjwal1/bert-tiny",
        freeze_base_model=True,
        # disable warmup to make sure the scheduler is not added which would set the learning rate
        # to 0
        warmup_proportion=0.0,
    )
    base_model_params = [param for name, param in model.base_model_named_parameters()]
    task_params = [param for name, param in model.task_named_parameters()]
    assert len(base_model_params) + len(task_params) == len(list(model.parameters()))
    for param in base_model_params:
        assert not param.requires_grad
    for param in task_params:
        assert param.requires_grad
