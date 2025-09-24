from typing import Dict

import pytest
import torch
from pytorch_lightning import Trainer
from torch import LongTensor
from torch.optim.lr_scheduler import LambdaLR
from transformers.modeling_outputs import SequenceClassifierOutput

from pytorch_ie.models import SequenceClassificationModelWithPooler
from pytorch_ie.models.sequence_classification_with_pooler import OutputType
from tests.models import trunc_number

NUM_CLASSES = 4
POOLER = "start_tokens"


@pytest.fixture
def inputs() -> Dict[str, LongTensor]:
    result_dict = {
        "input_ids": torch.tensor(
            [
                [
                    101,
                    28998,
                    13832,
                    3121,
                    2340,
                    138,
                    28996,
                    1759,
                    1120,
                    28999,
                    139,
                    28997,
                    119,
                    102,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    101,
                    1752,
                    5650,
                    119,
                    28998,
                    13832,
                    3121,
                    2340,
                    144,
                    28996,
                    1759,
                    1120,
                    28999,
                    145,
                    28997,
                    119,
                    1262,
                    1771,
                    146,
                    119,
                    102,
                    0,
                ],
                [
                    101,
                    1752,
                    5650,
                    119,
                    28998,
                    13832,
                    3121,
                    2340,
                    144,
                    28996,
                    1759,
                    1120,
                    145,
                    119,
                    1262,
                    1771,
                    28999,
                    146,
                    28997,
                    119,
                    102,
                    0,
                ],
                [
                    101,
                    1752,
                    5650,
                    119,
                    13832,
                    3121,
                    2340,
                    144,
                    1759,
                    1120,
                    28999,
                    145,
                    28997,
                    119,
                    1262,
                    1771,
                    28998,
                    146,
                    28996,
                    119,
                    102,
                    0,
                ],
                [
                    101,
                    1752,
                    5650,
                    119,
                    28998,
                    13832,
                    3121,
                    2340,
                    150,
                    28996,
                    1759,
                    1120,
                    28999,
                    151,
                    28997,
                    119,
                    1262,
                    1122,
                    1771,
                    152,
                    119,
                    102,
                ],
                [
                    101,
                    1752,
                    5650,
                    119,
                    13832,
                    3121,
                    2340,
                    150,
                    1759,
                    1120,
                    151,
                    119,
                    1262,
                    28998,
                    1122,
                    28996,
                    1771,
                    28999,
                    152,
                    28997,
                    119,
                    102,
                ],
                [
                    101,
                    1752,
                    5650,
                    119,
                    13832,
                    3121,
                    2340,
                    150,
                    1759,
                    1120,
                    151,
                    119,
                    1262,
                    28999,
                    1122,
                    28997,
                    1771,
                    28998,
                    152,
                    28996,
                    119,
                    102,
                ],
            ]
        ),
        "attention_mask": torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        ),
        "pooler_start_indices": torch.tensor(
            [[2, 10], [5, 13], [5, 17], [17, 11], [5, 13], [14, 18], [18, 14]]
        ),
        "pooler_end_indices": torch.tensor(
            [[6, 11], [9, 14], [9, 18], [18, 12], [9, 14], [15, 19], [19, 15]]
        ),
    }

    return result_dict  # type: ignore[return-value]


@pytest.fixture
def targets() -> Dict[str, LongTensor]:
    labels: LongTensor = torch.tensor([0, 1, 2, 3, 1, 2, 3])  # type: ignore[assignment]
    return {"labels": labels}


@pytest.fixture
def model() -> SequenceClassificationModelWithPooler:
    torch.manual_seed(42)
    result = SequenceClassificationModelWithPooler(
        model_name_or_path="prajjwal1/bert-tiny",
        num_classes=NUM_CLASSES,
        pooler=POOLER,
    )
    return result


def test_model(model):
    assert model is not None
    named_parameters = dict(model.named_parameters())
    parameter_means = {k: trunc_number(v.mean().item(), 7) for k, v in named_parameters.items()}
    parameter_means_expected = {
        "classifier.bias": -0.0253964,
        "classifier.weight": -0.000511,
        "model.embeddings.LayerNorm.bias": -0.0294608,
        "model.embeddings.LayerNorm.weight": 1.312345,
        "model.embeddings.position_embeddings.weight": 5.5e-05,
        "model.embeddings.token_type_embeddings.weight": -0.0015419,
        "model.embeddings.word_embeddings.weight": 0.0031152,
        "model.encoder.layer.0.attention.output.LayerNorm.bias": 0.0608714,
        "model.encoder.layer.0.attention.output.LayerNorm.weight": 1.199831,
        "model.encoder.layer.0.attention.output.dense.bias": 0.0007209,
        "model.encoder.layer.0.attention.output.dense.weight": 3.01e-05,
        "model.encoder.layer.0.attention.self.key.bias": 0.0020557,
        "model.encoder.layer.0.attention.self.key.weight": 0.0003863,
        "model.encoder.layer.0.attention.self.query.bias": 0.0185744,
        "model.encoder.layer.0.attention.self.query.weight": -0.0003949,
        "model.encoder.layer.0.attention.self.value.bias": 0.0065417,
        "model.encoder.layer.0.attention.self.value.weight": 4.22e-05,
        "model.encoder.layer.0.intermediate.dense.bias": -0.1219958,
        "model.encoder.layer.0.intermediate.dense.weight": -0.0011731,
        "model.encoder.layer.0.output.LayerNorm.bias": 0.005295,
        "model.encoder.layer.0.output.LayerNorm.weight": 1.2419648,
        "model.encoder.layer.0.output.dense.bias": -0.0013031,
        "model.encoder.layer.0.output.dense.weight": -0.0002212,
        "model.encoder.layer.1.attention.output.LayerNorm.bias": 0.0443237,
        "model.encoder.layer.1.attention.output.LayerNorm.weight": 1.0377343,
        "model.encoder.layer.1.attention.output.dense.bias": 0.0041446,
        "model.encoder.layer.1.attention.output.dense.weight": -2.43e-05,
        "model.encoder.layer.1.attention.self.key.bias": 0.0045062,
        "model.encoder.layer.1.attention.self.key.weight": 0.0001333,
        "model.encoder.layer.1.attention.self.query.bias": -0.0358397,
        "model.encoder.layer.1.attention.self.query.weight": -0.0007321,
        "model.encoder.layer.1.attention.self.value.bias": -0.0007094,
        "model.encoder.layer.1.attention.self.value.weight": 0.0001012,
        "model.encoder.layer.1.intermediate.dense.bias": -0.1247257,
        "model.encoder.layer.1.intermediate.dense.weight": -0.001344,
        "model.encoder.layer.1.output.LayerNorm.bias": -0.0474442,
        "model.encoder.layer.1.output.LayerNorm.weight": 1.017162,
        "model.encoder.layer.1.output.dense.bias": 0.000677,
        "model.encoder.layer.1.output.dense.weight": -5.32e-05,
        "model.pooler.dense.bias": -0.0052078,
        "model.pooler.dense.weight": 0.0001295,
        "pooler.pooler.missing_embeddings": 0.0630417,
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
    batch_size, seq_len = inputs["input_ids"].shape

    assert isinstance(model_output, SequenceClassifierOutput)

    logits = model_output.logits

    assert logits.shape == (batch_size, NUM_CLASSES)

    torch.testing.assert_close(
        logits,
        torch.tensor(
            [
                [
                    -0.29492446780204773,
                    -0.804599940776825,
                    -0.19659805297851562,
                    -1.0868580341339111,
                ],
                [
                    -0.3601434826850891,
                    -0.7454482316970825,
                    0.4882846474647522,
                    -1.0253472328186035,
                ],
                [
                    -0.40172430872917175,
                    -1.2119399309158325,
                    0.5856620669364929,
                    -1.0999149084091187,
                ],
                [
                    -0.09260234981775284,
                    -1.0742112398147583,
                    0.3299948275089264,
                    -0.5182554125785828,
                ],
                [
                    -0.40149545669555664,
                    -0.7731614708900452,
                    0.4616103768348694,
                    -1.0583568811416626,
                ],
                [
                    -0.14356234669685364,
                    -1.2634986639022827,
                    0.31660008430480957,
                    -0.7487461566925049,
                ],
                [
                    -0.11717557162046432,
                    -0.971996009349823,
                    0.3477852940559387,
                    -0.5993944406509399,
                ],
            ]
        ),
    )


def test_decode(model, model_output, inputs):
    decoded = model.decode(inputs=inputs, outputs=model_output)
    assert isinstance(decoded, dict)
    assert set(decoded) == {"labels", "probabilities"}
    labels = decoded["labels"]
    assert labels.shape == (inputs["input_ids"].shape[0],)
    torch.testing.assert_close(
        labels,
        torch.tensor([2, 2, 2, 2, 2, 2, 2]),
    )
    probabilities = decoded["probabilities"]
    assert probabilities.shape == (inputs["input_ids"].shape[0], NUM_CLASSES)
    torch.testing.assert_close(
        probabilities.round(decimals=4),
        torch.tensor(
            [
                [0.3168, 0.1903, 0.3495, 0.1435],
                [0.2207, 0.1502, 0.5156, 0.1135],
                [0.2161, 0.0961, 0.5802, 0.1075],
                [0.2814, 0.1054, 0.4294, 0.1838],
                [0.2184, 0.1506, 0.5177, 0.1132],
                [0.2893, 0.0944, 0.4583, 0.1580],
                [0.2751, 0.1170, 0.4380, 0.1699],
            ]
        ),
    )


def test_decode_with_multi_label(model_output, inputs):
    torch.manual_seed(42)
    model = SequenceClassificationModelWithPooler(
        model_name_or_path="prajjwal1/bert-tiny",
        num_classes=NUM_CLASSES,
        pooler=POOLER,
        multi_label=True,
    )
    decoded = model.decode(inputs=inputs, outputs=model_output)
    assert isinstance(decoded, dict)
    assert set(decoded) == {"labels", "probabilities"}
    labels = decoded["labels"]
    assert labels.shape == (inputs["input_ids"].shape[0], NUM_CLASSES)
    torch.testing.assert_close(
        labels,
        torch.tensor(
            [
                [0, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 0],
            ]
        ),
    )
    probabilities = decoded["probabilities"]
    assert probabilities.shape == (inputs["input_ids"].shape[0], NUM_CLASSES)
    torch.testing.assert_close(
        probabilities.round(decimals=4),
        torch.tensor(
            [
                [0.4268, 0.3090, 0.4510, 0.2522],
                [0.4109, 0.3218, 0.6197, 0.2640],
                [0.4009, 0.2294, 0.6424, 0.2498],
                [0.4769, 0.2546, 0.5818, 0.3733],
                [0.4010, 0.3158, 0.6134, 0.2576],
                [0.4642, 0.2204, 0.5785, 0.3211],
                [0.4707, 0.2745, 0.5861, 0.3545],
            ]
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
    torch.testing.assert_close(loss, torch.tensor(1.3899686336517334))


def test_validation_step(batch, model):
    # set the seed to make sure the loss is deterministic
    torch.manual_seed(42)
    loss = model.validation_step(batch, batch_idx=0)
    assert loss is not None
    torch.testing.assert_close(loss, torch.tensor(1.3899686336517334))


def test_test_step(batch, model):
    # set the seed to make sure the loss is deterministic
    torch.manual_seed(42)
    loss = model.test_step(batch, batch_idx=0)
    assert loss is not None
    torch.testing.assert_close(loss, torch.tensor(1.3899686336517334))


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
        "classifier.weight",
        "pooler.pooler.missing_embeddings",
        "classifier.bias",
    }


def test_configure_optimizers_with_warmup():
    model = SequenceClassificationModelWithPooler(
        model_name_or_path="prajjwal1/bert-tiny",
        num_classes=NUM_CLASSES,
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
    model = SequenceClassificationModelWithPooler(
        model_name_or_path="prajjwal1/bert-tiny",
        num_classes=NUM_CLASSES,
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
    # classifier head parameters
    param_group = optimizer.param_groups[1]
    assert len(param_group["params"]) == 2
    assert param_group["lr"] == 1e-3
    # ensure that all parameters are covered
    assert set(optimizer.param_groups[0]["params"] + optimizer.param_groups[1]["params"]) == set(
        model.parameters()
    )


def test_freeze_base_model(monkeypatch, inputs, targets):
    model = SequenceClassificationModelWithPooler(
        model_name_or_path="prajjwal1/bert-tiny",
        num_classes=NUM_CLASSES,
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
