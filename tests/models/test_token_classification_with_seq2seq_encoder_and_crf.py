import pytest
import torch
from pytorch_lightning import Trainer

from pytorch_ie.models import TokenClassificationModelWithSeq2SeqEncoderAndCrf
from pytorch_ie.models.common import TESTING, TRAINING, VALIDATION
from pytorch_ie.taskmodules import LabeledSpanExtractionByTokenClassificationTaskModule
from tests import _config_to_str
from tests.models import trunc_number

CONFIGS = [{}, {"use_crf": False}]
CONFIG_DICT = {_config_to_str(cfg): cfg for cfg in CONFIGS}


@pytest.fixture(scope="module", params=CONFIG_DICT.keys())
def config_str(request):
    return request.param


@pytest.fixture(scope="module")
def config(config_str):
    return CONFIG_DICT[config_str]


@pytest.fixture
def taskmodule_config():
    return {
        "taskmodule_type": "LabeledSpanExtractionByTokenClassificationTaskModule",
        "tokenizer_name_or_path": "bert-base-cased",
        "span_annotation": "entities",
        "partition_annotation": None,
        "label_pad_id": -100,
        "labels": ["ORG", "PER"],
        "include_ill_formed_predictions": True,
        "combine_token_scores_method": "mean",
        "tokenize_kwargs": None,
        "pad_kwargs": None,
        "log_precision_recall_metrics": True,
    }


def test_taskmodule_config(documents, taskmodule_config):
    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = LabeledSpanExtractionByTokenClassificationTaskModule(
        span_annotation="entities",
        tokenizer_name_or_path=tokenizer_name_or_path,
    )
    taskmodule.prepare(documents)
    assert taskmodule.config == taskmodule_config


def test_batch(documents, batch, taskmodule_config):
    taskmodule = LabeledSpanExtractionByTokenClassificationTaskModule.from_config(
        taskmodule_config
    )
    encodings = taskmodule.encode(documents, encode_target=True)
    batch_from_documents = taskmodule.collate(encodings[:4])

    inputs, targets = batch
    inputs_from_documents, targets_from_documents = batch_from_documents
    torch.testing.assert_close(inputs["input_ids"], inputs_from_documents["input_ids"])
    torch.testing.assert_close(inputs["attention_mask"], inputs_from_documents["attention_mask"])
    torch.testing.assert_close(targets, targets_from_documents)


@pytest.fixture
def batch():
    inputs = {
        "input_ids": torch.tensor(
            [
                [101, 138, 1423, 5650, 119, 102, 0, 0, 0, 0, 0, 0],
                [101, 13832, 3121, 2340, 138, 1759, 1120, 139, 119, 102, 0, 0],
                [101, 13832, 3121, 2340, 140, 1105, 141, 119, 102, 0, 0, 0],
                [101, 1752, 5650, 119, 13832, 3121, 2340, 142, 1105, 143, 119, 102],
            ]
        ).to(torch.long),
        "attention_mask": torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        ),
        "special_tokens_mask": torch.tensor(
            [
                [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        ),
    }
    targets = {
        "labels": torch.tensor(
            [
                [-100, 0, 0, 0, 0, -100, -100, -100, -100, -100, -100, -100],
                [-100, 3, 4, 4, 4, 0, 0, 1, 0, -100, -100, -100],
                [-100, 3, 4, 4, 4, 0, 1, 0, -100, -100, -100, -100],
                [-100, 0, 0, 0, 3, 4, 4, 4, 0, 1, 0, -100],
            ]
        )
    }
    return inputs, targets


@pytest.fixture
def model(batch, config, taskmodule_config) -> TokenClassificationModelWithSeq2SeqEncoderAndCrf:
    seq2seq_dict = {
        "type": "linear",
        "out_features": 10,
    }
    torch.manual_seed(42)
    model = TokenClassificationModelWithSeq2SeqEncoderAndCrf(
        model_name_or_path="prajjwal1/bert-tiny",
        num_classes=5,
        seq2seq_encoder=seq2seq_dict,
        taskmodule_config=taskmodule_config,
        metric_stages=["val", "test"],
        **config,
    )
    return model


def test_model(model, config):
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
        "seq2seq_encoder.weight": -0.0015382,
        "seq2seq_encoder.bias": -0.0105704,
        "classifier.weight": 0.0261459,
        "classifier.bias": -0.0157966,
    }
    if config.get("use_crf", True):
        parameter_means_expected.update(
            {
                "crf.start_transitions": -0.0341042,
                "crf.end_transitions": 0.0140624,
                "crf.transitions": 0.0056733,
            }
        )
    assert parameter_means == parameter_means_expected


def test_model_pickleable(model):
    import pickle

    pickle.dumps(model)


def test_freeze_base_model():
    model = TokenClassificationModelWithSeq2SeqEncoderAndCrf(
        model_name_or_path="prajjwal1/bert-tiny",
        num_classes=5,
        freeze_base_model=True,
    )

    base_model_params = dict(model.model.named_parameters(prefix="model"))
    assert len(base_model_params) > 0
    for param in base_model_params.values():
        assert not param.requires_grad
    task_params = {
        name: param for name, param in model.named_parameters() if name not in base_model_params
    }
    assert len(task_params) > 0
    for param in task_params.values():
        assert param.requires_grad


def test_tune_base_model():
    model = TokenClassificationModelWithSeq2SeqEncoderAndCrf(
        model_name_or_path="prajjwal1/bert-tiny",
        num_classes=5,
    )
    base_model_params = dict(model.model.named_parameters(prefix="model"))
    assert len(base_model_params) > 0
    for param in base_model_params.values():
        assert param.requires_grad
    task_params = {
        name: param for name, param in model.named_parameters() if name not in base_model_params
    }
    assert len(task_params) > 0
    for param in task_params.values():
        assert param.requires_grad


def test_forward(batch, model):
    inputs, targets = batch
    batch_size, seq_len = inputs["input_ids"].shape
    num_classes = int(torch.max(targets["labels"]) + 1)

    # set seed to make sure the output is deterministic
    torch.manual_seed(42)
    output = model.forward(inputs)
    assert set(output) == {"logits"}
    logits = output["logits"]
    assert logits.shape == (batch_size, seq_len, num_classes)
    # check the first batch entry
    torch.testing.assert_close(
        logits[0],
        torch.tensor(
            [
                [
                    -1.065280795097351,
                    0.22260898351669312,
                    -0.013371739536523819,
                    1.0213487148284912,
                    -0.08737741410732269,
                ],
                [
                    -1.092915415763855,
                    0.07986105978488922,
                    0.011286348104476929,
                    0.7147902250289917,
                    -0.014343257993459702,
                ],
                [
                    -1.0107779502868652,
                    0.2041827142238617,
                    -0.06531291455030441,
                    0.6551182270050049,
                    0.04944971576333046,
                ],
                [
                    -0.3324984312057495,
                    0.27757787704467773,
                    0.13295423984527588,
                    0.26407280564308167,
                    -0.007371138781309128,
                ],
                [
                    -0.6176304817199707,
                    0.12915551662445068,
                    0.268213152885437,
                    0.43618908524513245,
                    -0.13303528726100922,
                ],
                [
                    -0.5220450758934021,
                    0.37291139364242554,
                    0.2522115111351013,
                    0.7383102178573608,
                    0.1278681606054306,
                ],
                [
                    -1.0737248659133911,
                    0.0029090046882629395,
                    0.06924695521593094,
                    0.6680881977081299,
                    -0.15523286163806915,
                ],
                [
                    -0.5176048278808594,
                    -0.01018303632736206,
                    0.14543311297893524,
                    0.5191693305969238,
                    -0.3461107611656189,
                ],
                [
                    -0.9277648329734802,
                    0.3154565095901489,
                    -0.07648143172264099,
                    0.4210910201072693,
                    0.2663896083831787,
                ],
                [
                    -0.8864655494689941,
                    0.2862459421157837,
                    -0.04168111830949783,
                    0.4992614984512329,
                    0.28455498814582825,
                ],
                [
                    -0.9500657916069031,
                    0.1869449019432068,
                    -0.005329027771949768,
                    0.5908203721046448,
                    0.06730394065380096,
                ],
                [
                    -0.5336291193962097,
                    -0.053214408457279205,
                    0.22038350999355316,
                    0.48135989904403687,
                    -0.4338146448135376,
                ],
            ]
        ),
    )

    # check the sums per sequence
    torch.testing.assert_close(
        logits.sum(1),
        torch.tensor(
            [
                [
                    -9.530403137207031,
                    2.0144565105438232,
                    0.8975526690483093,
                    7.009620189666748,
                    -0.3817189633846283,
                ],
                [
                    -4.351415634155273,
                    0.3694552183151245,
                    -0.8337129354476929,
                    3.612205743789673,
                    0.15454095602035522,
                ],
                [
                    -6.173098564147949,
                    -2.6261491775512695,
                    0.47521746158599854,
                    3.344158172607422,
                    -5.086399078369141,
                ],
                [
                    -9.28173542022705,
                    -1.6196215152740479,
                    0.18393829464912415,
                    5.492751121520996,
                    -4.148656845092773,
                ],
            ]
        ),
    )


def test_step(batch, model, config):
    torch.manual_seed(42)
    loss = model._step("train", batch)
    assert loss is not None
    if config == {}:
        torch.testing.assert_close(loss, torch.tensor(75.52511596679688))
    elif config == {"use_crf": False}:
        torch.testing.assert_close(loss, torch.tensor(1.9434731006622314))
    else:
        raise ValueError(f"Unknown config: {config}")


def test_training_step_and_on_epoch_end(batch, model, config):
    metric = model._get_metric(TRAINING, batch_idx=0)
    assert metric is None
    loss = model.training_step(batch, batch_idx=0)
    assert loss is not None
    if config == {}:
        torch.testing.assert_close(loss, torch.tensor(77.59623718261719))
    elif config == {"use_crf": False}:
        torch.testing.assert_close(loss, torch.tensor(1.9865683317184448))
    else:
        raise ValueError(f"Unknown config: {config}")

    model.on_train_epoch_end()


def test_training_step_without_attention_mask(batch, model, config):
    inputs, targets = batch
    inputs_without_attention_mask = {k: v for k, v in inputs.items() if k != "attention_mask"}
    loss = model.training_step(batch=(inputs_without_attention_mask, targets), batch_idx=0)
    assert loss is not None
    if config == {}:
        torch.testing.assert_close(loss, torch.tensor(103.0061264038086))
    elif config == {"use_crf": False}:
        torch.testing.assert_close(loss, torch.tensor(1.9988830089569092))
    else:
        raise ValueError(f"Unknown config: {config}")


def test_validation_step_and_on_epoch_end(batch, model, config):
    metric = model._get_metric(VALIDATION, batch_idx=0)
    metric.reset()
    loss = model.validation_step(batch, batch_idx=0)
    assert loss is not None
    metric_values = {k: v.item() for k, v in metric.compute().items()}
    if config == {}:
        torch.testing.assert_close(loss, torch.tensor(77.59623718261719))
        assert metric_values == {
            "token/macro/f1": 0.20666667819023132,
            "token/micro/f1": 0.2068965584039688,
            "token/macro/precision": 0.29019609093666077,
            "token/macro/recall": 0.2666666805744171,
            "token/micro/precision": 0.2068965584039688,
            "token/micro/recall": 0.2068965584039688,
            "span/ORG/f1": 0.3636363744735718,
            "span/ORG/recall": 0.25,
            "span/ORG/precision": 0.6666666865348816,
            "span/PER/f1": 0.0,
            "span/PER/recall": 0.0,
            "span/PER/precision": 0.0,
            "span/micro/f1": 0.12121212482452393,
            "span/micro/recall": 0.07407407462596893,
            "span/micro/precision": 0.3333333432674408,
            "span/macro/f1": 0.1818181872367859,
            "span/macro/recall": 0.125,
            "span/macro/precision": 0.3333333432674408,
        }
    elif config == {"use_crf": False}:
        torch.testing.assert_close(loss, torch.tensor(1.9865683317184448))
        assert metric_values == {
            "token/macro/f1": 0.11717171967029572,
            "token/micro/f1": 0.17241379618644714,
            "token/macro/precision": 0.22500000894069672,
            "token/macro/recall": 0.24444444477558136,
            "token/micro/precision": 0.17241379618644714,
            "token/micro/recall": 0.17241379618644714,
            "span/ORG/f1": 0.0,
            "span/ORG/recall": 0.0,
            "span/ORG/precision": 0.0,
            "span/PER/f1": 0.0,
            "span/PER/recall": 0.0,
            "span/PER/precision": 0.0,
            "span/micro/f1": 0.0,
            "span/micro/recall": 0.0,
            "span/micro/precision": 0.0,
            "span/macro/f1": 0.0,
            "span/macro/recall": 0.0,
            "span/macro/precision": 0.0,
        }
    else:
        raise ValueError(f"Unknown config: {config}")

    model.on_validation_epoch_end()


def test_test_step_and_on_epoch_end(batch, model, config):
    metric = model._get_metric(TESTING, batch_idx=0)
    metric.reset()
    loss = model.test_step(batch, batch_idx=0)
    assert loss is not None
    metric_values = {k: v.item() for k, v in metric.compute().items()}
    if config == {}:
        torch.testing.assert_close(loss, torch.tensor(77.59623718261719))
        assert metric_values == {
            "token/macro/f1": 0.20666667819023132,
            "token/micro/f1": 0.2068965584039688,
            "token/macro/precision": 0.29019609093666077,
            "token/macro/recall": 0.2666666805744171,
            "token/micro/precision": 0.2068965584039688,
            "token/micro/recall": 0.2068965584039688,
            "span/ORG/f1": 0.3636363744735718,
            "span/ORG/recall": 0.25,
            "span/ORG/precision": 0.6666666865348816,
            "span/PER/f1": 0.0,
            "span/PER/recall": 0.0,
            "span/PER/precision": 0.0,
            "span/micro/f1": 0.12121212482452393,
            "span/micro/recall": 0.07407407462596893,
            "span/micro/precision": 0.3333333432674408,
            "span/macro/f1": 0.1818181872367859,
            "span/macro/recall": 0.125,
            "span/macro/precision": 0.3333333432674408,
        }
    elif config == {"use_crf": False}:
        torch.testing.assert_close(loss, torch.tensor(1.9865683317184448))
        assert metric_values == {
            "token/macro/f1": 0.11717171967029572,
            "token/micro/f1": 0.17241379618644714,
            "token/macro/precision": 0.22500000894069672,
            "token/macro/recall": 0.24444444477558136,
            "token/micro/precision": 0.17241379618644714,
            "token/micro/recall": 0.17241379618644714,
            "span/ORG/f1": 0.0,
            "span/ORG/recall": 0.0,
            "span/ORG/precision": 0.0,
            "span/PER/f1": 0.0,
            "span/PER/recall": 0.0,
            "span/PER/precision": 0.0,
            "span/micro/f1": 0.0,
            "span/micro/recall": 0.0,
            "span/micro/precision": 0.0,
            "span/macro/f1": 0.0,
            "span/macro/recall": 0.0,
            "span/macro/precision": 0.0,
        }
    else:
        raise ValueError(f"Unknown config: {config}")

    model.on_test_epoch_end()


@pytest.mark.parametrize("test_step", [False, True])
def test_predict_and_predict_step(model, batch, config, test_step):
    torch.manual_seed(42)
    if test_step:
        predictions = model.predict_step(batch, batch_idx=0, dataloader_idx=0)
    else:
        predictions = model.predict(batch[0])

    assert set(predictions) == {"labels", "probabilities"}
    labels = predictions["labels"]
    probabilities = predictions["probabilities"]
    if config == {}:
        torch.testing.assert_close(
            labels,
            torch.tensor(
                [
                    [-100, 3, 3, 1, 3, -100, -100, -100, -100, -100, -100, -100],
                    [-100, 3, 1, 4, 4, 3, 3, 3, 2, -100, -100, -100],
                    [-100, 3, 2, 2, 3, 3, 3, 2, -100, -100, -100, -100],
                    [-100, 3, 3, 3, 2, 3, 1, 4, 3, 3, 2, -100],
                ]
            ),
        )
    elif config == {"use_crf": False}:
        torch.testing.assert_close(
            labels,
            torch.tensor(
                [
                    [-100, 3, 3, 1, 3, -100, -100, -100, -100, -100, -100, -100],
                    [-100, 3, 3, 4, 4, 3, 3, 3, 3, -100, -100, -100],
                    [-100, 3, 2, 2, 3, 3, 3, 2, -100, -100, -100, -100],
                    [-100, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, -100],
                ]
            ),
        )
    else:
        raise ValueError(f"Unknown config: {config}")

    assert labels.shape == batch[1]["labels"].shape
    torch.testing.assert_close(
        probabilities[:2].round(decimals=4),
        torch.tensor(
            [
                [
                    [0.0549, 0.1991, 0.1573, 0.4426, 0.1461],
                    [0.0614, 0.1984, 0.1853, 0.3744, 0.1806],
                    [0.0661, 0.2229, 0.1702, 0.3499, 0.1909],
                    [0.1310, 0.2411, 0.2087, 0.2379, 0.1813],
                    [0.0997, 0.2104, 0.2418, 0.2861, 0.1619],
                    [0.0904, 0.2213, 0.1961, 0.3189, 0.1732],
                    [0.0654, 0.1920, 0.2052, 0.3734, 0.1639],
                    [0.1162, 0.1929, 0.2254, 0.3276, 0.1379],
                    [0.0716, 0.2483, 0.1678, 0.2759, 0.2364],
                    [0.0726, 0.2344, 0.1689, 0.2901, 0.2340],
                    [0.0708, 0.2207, 0.1821, 0.3305, 0.1958],
                    [0.1162, 0.1879, 0.2470, 0.3206, 0.1284],
                ],
                [
                    [0.1242, 0.1911, 0.1516, 0.3256, 0.2075],
                    [0.1291, 0.2089, 0.2046, 0.2890, 0.1684],
                    [0.2033, 0.2016, 0.1920, 0.2260, 0.1771],
                    [0.1793, 0.2191, 0.1800, 0.1889, 0.2328],
                    [0.1854, 0.2150, 0.1638, 0.1898, 0.2460],
                    [0.1363, 0.2007, 0.1738, 0.2887, 0.2005],
                    [0.1254, 0.2014, 0.1826, 0.2890, 0.2016],
                    [0.1305, 0.2056, 0.2056, 0.2590, 0.1993],
                    [0.1400, 0.2022, 0.2252, 0.2544, 0.1783],
                    [0.1299, 0.2051, 0.1933, 0.2751, 0.1966],
                    [0.1088, 0.2086, 0.1599, 0.2861, 0.2367],
                    [0.0910, 0.1794, 0.1840, 0.3793, 0.1663],
                ],
            ]
        ),
    )


def test_configure_optimizers(model):
    model.trainer = Trainer(max_epochs=10)
    optimizer_and_schedular = model.configure_optimizers()
    assert optimizer_and_schedular is not None
    optimizers, schedulers = optimizer_and_schedular

    assert len(optimizers) == 1
    optimizer = optimizers[0]
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["lr"] == 1e-05
    assert optimizer.defaults["weight_decay"] == 0.01
    assert optimizer.defaults["eps"] == 1e-08

    assert len(schedulers) == 1
    scheduler = schedulers[0]
    assert isinstance(scheduler["scheduler"], torch.optim.lr_scheduler.LambdaLR)


def test_configure_optimizers_with_task_learning_rate():
    model = TokenClassificationModelWithSeq2SeqEncoderAndCrf(
        model_name_or_path="prajjwal1/bert-tiny",
        num_classes=5,
        warmup_proportion=0.0,
        task_learning_rate=1e-4,
    )
    optimizer = model.configure_optimizers()
    assert optimizer is not None
    assert isinstance(optimizer, torch.optim.AdamW)
    assert len(optimizer.param_groups) == 2
    # check that all parameters are in the optimizer
    assert set(optimizer.param_groups[0]["params"]) | set(
        optimizer.param_groups[1]["params"]
    ) == set(model.parameters())

    # base model parameters
    param_group = optimizer.param_groups[0]
    assert param_group["lr"] == 1e-05
    assert len(param_group["params"]) == 39

    # task parameters
    param_group = optimizer.param_groups[1]
    assert param_group["lr"] == 1e-04
    assert len(param_group["params"]) == 5
