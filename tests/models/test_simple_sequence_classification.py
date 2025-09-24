from typing import Dict

import pytest
import torch
from pytorch_lightning import Trainer
from torch.optim.lr_scheduler import LambdaLR
from transformers.modeling_outputs import SequenceClassifierOutput

from pytorch_ie.models import SimpleSequenceClassificationModel
from pytorch_ie.models.simple_sequence_classification import OutputType
from tests.models import trunc_number

NUM_CLASSES = 4


@pytest.fixture
def inputs() -> Dict[str, torch.LongTensor]:
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
    }

    return result_dict  # type: ignore[return-value]


@pytest.fixture
def targets() -> Dict[str, torch.LongTensor]:
    labels: torch.LongTensor = torch.tensor([0, 1, 2, 3, 1, 2, 3])  # type: ignore[assignment]
    return {"labels": labels}


@pytest.fixture
def model() -> SimpleSequenceClassificationModel:
    torch.manual_seed(42)
    result = SimpleSequenceClassificationModel(
        model_name_or_path="prajjwal1/bert-tiny",
        num_classes=NUM_CLASSES,
    )
    return result


def test_model(model):
    assert model is not None
    named_parameters = dict(model.named_parameters())
    parameter_means = {k: trunc_number(v.mean().item(), 7) for k, v in named_parameters.items()}
    parameter_means_expected = {
        "model.bert.embeddings.word_embeddings.weight": 0.0031152,
        "model.bert.embeddings.position_embeddings.weight": 5.5e-05,
        "model.bert.embeddings.token_type_embeddings.weight": -0.0015419,
        "model.bert.embeddings.LayerNorm.weight": 1.312345,
        "model.bert.embeddings.LayerNorm.bias": -0.0294608,
        "model.bert.encoder.layer.0.attention.self.query.weight": -0.0003949,
        "model.bert.encoder.layer.0.attention.self.query.bias": 0.0185744,
        "model.bert.encoder.layer.0.attention.self.key.weight": 0.0003863,
        "model.bert.encoder.layer.0.attention.self.key.bias": 0.0020557,
        "model.bert.encoder.layer.0.attention.self.value.weight": 4.22e-05,
        "model.bert.encoder.layer.0.attention.self.value.bias": 0.0065417,
        "model.bert.encoder.layer.0.attention.output.dense.weight": 3.01e-05,
        "model.bert.encoder.layer.0.attention.output.dense.bias": 0.0007209,
        "model.bert.encoder.layer.0.attention.output.LayerNorm.weight": 1.199831,
        "model.bert.encoder.layer.0.attention.output.LayerNorm.bias": 0.0608714,
        "model.bert.encoder.layer.0.intermediate.dense.weight": -0.0011731,
        "model.bert.encoder.layer.0.intermediate.dense.bias": -0.1219958,
        "model.bert.encoder.layer.0.output.dense.weight": -0.0002212,
        "model.bert.encoder.layer.0.output.dense.bias": -0.0013031,
        "model.bert.encoder.layer.0.output.LayerNorm.weight": 1.2419648,
        "model.bert.encoder.layer.0.output.LayerNorm.bias": 0.005295,
        "model.bert.encoder.layer.1.attention.self.query.weight": -0.0007321,
        "model.bert.encoder.layer.1.attention.self.query.bias": -0.0358397,
        "model.bert.encoder.layer.1.attention.self.key.weight": 0.0001333,
        "model.bert.encoder.layer.1.attention.self.key.bias": 0.0045062,
        "model.bert.encoder.layer.1.attention.self.value.weight": 0.0001012,
        "model.bert.encoder.layer.1.attention.self.value.bias": -0.0007094,
        "model.bert.encoder.layer.1.attention.output.dense.weight": -2.43e-05,
        "model.bert.encoder.layer.1.attention.output.dense.bias": 0.0041446,
        "model.bert.encoder.layer.1.attention.output.LayerNorm.weight": 1.0377343,
        "model.bert.encoder.layer.1.attention.output.LayerNorm.bias": 0.0443237,
        "model.bert.encoder.layer.1.intermediate.dense.weight": -0.001344,
        "model.bert.encoder.layer.1.intermediate.dense.bias": -0.1247257,
        "model.bert.encoder.layer.1.output.dense.weight": -5.32e-05,
        "model.bert.encoder.layer.1.output.dense.bias": 0.000677,
        "model.bert.encoder.layer.1.output.LayerNorm.weight": 1.017162,
        "model.bert.encoder.layer.1.output.LayerNorm.bias": -0.0474442,
        "model.bert.pooler.dense.weight": 0.0001295,
        "model.bert.pooler.dense.bias": -0.0052078,
        "model.classifier.weight": 0.0005538,
        "model.classifier.bias": 0.0,
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


def test_forward(model_output, inputs):
    batch_size = inputs["input_ids"].shape[0]
    assert isinstance(model_output, SequenceClassifierOutput)
    assert set(model_output) == {"logits"}
    logits = model_output["logits"]

    assert logits.shape == (batch_size, NUM_CLASSES)

    torch.testing.assert_close(
        logits,
        torch.tensor(
            [
                [
                    0.16545572876930237,
                    0.17544983327388763,
                    -0.011048287153244019,
                    0.05337674915790558,
                ],
                [
                    0.14748695492744446,
                    0.16249355673789978,
                    -0.058017998933792114,
                    0.025398850440979004,
                ],
                [
                    0.14271709322929382,
                    0.16188383102416992,
                    -0.061113521456718445,
                    0.026494741439819336,
                ],
                [
                    0.15641027688980103,
                    0.17225395143032074,
                    -0.05567866563796997,
                    0.022433891892433167,
                ],
                [
                    0.15785054862499237,
                    0.16935551166534424,
                    -0.054724469780921936,
                    0.012338697910308838,
                ],
                [
                    0.16152460873126984,
                    0.17789196968078613,
                    -0.053754448890686035,
                    0.008724510669708252,
                ],
                [
                    0.16836002469062805,
                    0.17842254042625427,
                    -0.052499815821647644,
                    0.006823211908340454,
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
        torch.tensor([1, 1, 1, 1, 1, 1, 1]),
    )
    probabilities = decoded["probabilities"]
    assert probabilities.shape == (inputs["input_ids"].shape[0], NUM_CLASSES)
    torch.testing.assert_close(
        probabilities,
        torch.tensor(
            [
                [
                    0.2672215402126312,
                    0.26990556716918945,
                    0.22398385405540466,
                    0.23888900876045227,
                ],
                [
                    0.26922059059143066,
                    0.27329114079475403,
                    0.21920911967754364,
                    0.23827917873859406,
                ],
                [0.2684398889541626, 0.2736346125602722, 0.21893969178199768, 0.23898591101169586],
                [0.2703087329864502, 0.2746255099773407, 0.21865077316761017, 0.23641489446163177],
                [0.2713961601257324, 0.2745365798473358, 0.21942369639873505, 0.2346435934305191],
                [
                    0.27165648341178894,
                    0.27613937854766846,
                    0.21904107928276062,
                    0.23316311836242676,
                ],
                [0.2730168402194977, 0.2757779359817505, 0.21891282498836517, 0.23229233920574188],
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
    torch.testing.assert_close(loss, torch.tensor(1.4069921970367432))


def test_validation_step(batch, model):
    # set the seed to make sure the loss is deterministic
    torch.manual_seed(42)
    loss = model.validation_step(batch, batch_idx=0)
    assert loss is not None
    torch.testing.assert_close(loss, torch.tensor(1.4069921970367432))


def test_test_step(batch, model):
    # set the seed to make sure the loss is deterministic
    torch.manual_seed(42)
    loss = model.test_step(batch, batch_idx=0)
    assert loss is not None
    torch.testing.assert_close(loss, torch.tensor(1.4069921970367432))


def test_base_model_named_parameters(model):
    base_model_named_parameters = dict(model.base_model_named_parameters())
    assert set(base_model_named_parameters) == {
        "model.bert.pooler.dense.bias",
        "model.bert.encoder.layer.0.intermediate.dense.weight",
        "model.bert.encoder.layer.0.intermediate.dense.bias",
        "model.bert.encoder.layer.1.attention.output.dense.weight",
        "model.bert.encoder.layer.1.attention.output.LayerNorm.weight",
        "model.bert.encoder.layer.1.attention.self.query.weight",
        "model.bert.encoder.layer.1.output.dense.weight",
        "model.bert.encoder.layer.0.output.dense.bias",
        "model.bert.encoder.layer.1.intermediate.dense.bias",
        "model.bert.encoder.layer.1.attention.self.value.bias",
        "model.bert.encoder.layer.0.attention.output.dense.weight",
        "model.bert.encoder.layer.0.attention.self.query.bias",
        "model.bert.encoder.layer.0.attention.self.value.bias",
        "model.bert.encoder.layer.1.output.dense.bias",
        "model.bert.encoder.layer.1.attention.self.query.bias",
        "model.bert.encoder.layer.1.attention.output.LayerNorm.bias",
        "model.bert.encoder.layer.0.attention.self.query.weight",
        "model.bert.encoder.layer.0.attention.output.LayerNorm.bias",
        "model.bert.encoder.layer.0.attention.self.key.bias",
        "model.bert.encoder.layer.1.intermediate.dense.weight",
        "model.bert.encoder.layer.1.output.LayerNorm.bias",
        "model.bert.encoder.layer.1.output.LayerNorm.weight",
        "model.bert.encoder.layer.0.attention.self.key.weight",
        "model.bert.encoder.layer.1.attention.output.dense.bias",
        "model.bert.encoder.layer.0.attention.output.dense.bias",
        "model.bert.embeddings.LayerNorm.bias",
        "model.bert.encoder.layer.0.attention.self.value.weight",
        "model.bert.encoder.layer.0.attention.output.LayerNorm.weight",
        "model.bert.embeddings.token_type_embeddings.weight",
        "model.bert.encoder.layer.0.output.LayerNorm.weight",
        "model.bert.embeddings.position_embeddings.weight",
        "model.bert.encoder.layer.1.attention.self.key.bias",
        "model.bert.embeddings.LayerNorm.weight",
        "model.bert.encoder.layer.0.output.LayerNorm.bias",
        "model.bert.encoder.layer.1.attention.self.key.weight",
        "model.bert.pooler.dense.weight",
        "model.bert.encoder.layer.0.output.dense.weight",
        "model.bert.embeddings.word_embeddings.weight",
        "model.bert.encoder.layer.1.attention.self.value.weight",
    }


def test_task_named_parameters(model):
    task_named_parameters = dict(model.task_named_parameters())
    assert set(task_named_parameters) == {
        "model.classifier.weight",
        "model.classifier.bias",
    }


def test_configure_optimizers_with_warmup():
    model = SimpleSequenceClassificationModel(
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
    model = SimpleSequenceClassificationModel(
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
    model = SimpleSequenceClassificationModel(
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


def test_base_model_named_parameters_wrong_prefix(monkeypatch):
    model = SimpleSequenceClassificationModel(
        model_name_or_path="prajjwal1/bert-tiny",
        num_classes=NUM_CLASSES,
        base_model_prefix="wrong_prefix",
    )
    with pytest.raises(ValueError) as excinfo:
        model.base_model_named_parameters()
    assert (
        str(excinfo.value)
        == "Base model with prefix 'wrong_prefix' not found in BertForSequenceClassification"
    )
