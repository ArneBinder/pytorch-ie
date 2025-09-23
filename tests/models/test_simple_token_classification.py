from typing import Any, Dict, List

import pytest
import torch

from pytorch_ie.models import SimpleTokenClassificationModel
from pytorch_ie.models.common import TESTING, TRAINING, VALIDATION
from pytorch_ie.taskmodules import LabeledSpanExtractionByTokenClassificationTaskModule
from tests import _config_to_str
from tests.models import trunc_number

CONFIGS: List[Dict[str, Any]] = [{}]
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
        "tokenize_kwargs": None,
        "pad_kwargs": None,
        "combine_token_scores_method": "mean",
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
    # just take the first 4 encodings
    batch_from_documents = taskmodule.collate(encodings[:4])

    inputs, targets = batch
    inputs_from_documents, targets_from_documents = batch_from_documents
    assert set(inputs) == set(inputs_from_documents)
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
def model(monkeypatch, batch, config, taskmodule_config) -> SimpleTokenClassificationModel:
    torch.manual_seed(42)
    model = SimpleTokenClassificationModel(
        model_name_or_path="prajjwal1/bert-tiny",
        num_classes=5,
        taskmodule_config=taskmodule_config,
        metric_stages=["val", "test"],
    )
    return model


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
        "model.classifier.weight": 0.0005138,
        "model.classifier.bias": 0.0,
    }
    assert parameter_means == parameter_means_expected


def test_model_pickleable(model):
    import pickle

    pickle.dumps(model)


def test_forward(batch, model):
    inputs, targets = batch
    batch_size, seq_len = inputs["input_ids"].shape
    num_classes = model.config["num_classes"]

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
                    -0.13442197442054749,
                    -0.06983129680156708,
                    0.17513807117938995,
                    -0.24002864956855774,
                    0.08871676027774811,
                ],
                [
                    -0.032687313854694366,
                    -0.2071131318807602,
                    0.10695032775402069,
                    -0.05829116329550743,
                    -0.21174949407577515,
                ],
                [
                    -0.17153336107730865,
                    -0.2230629324913025,
                    -0.11457862704992294,
                    0.03658870607614517,
                    -0.242639422416687,
                ],
                [
                    -0.07552017271518707,
                    -0.20950022339820862,
                    0.041016221046447754,
                    -0.13453879952430725,
                    -0.09942213445901871,
                ],
                [
                    -0.19299760460853577,
                    -0.2081824392080307,
                    0.20880958437919617,
                    -0.028745755553245544,
                    -0.14375154674053192,
                ],
                [
                    -0.20548884570598602,
                    -0.17012161016464233,
                    0.0647551566362381,
                    -0.090476393699646,
                    -0.1362220048904419,
                ],
                [
                    -0.09553629904985428,
                    -0.1303575187921524,
                    0.2995688021183014,
                    -0.04689876735210419,
                    -0.17737819254398346,
                ],
                [
                    -0.030023209750652313,
                    -0.12308696657419205,
                    0.2582213580608368,
                    -0.04085375368595123,
                    -0.16487300395965576,
                ],
                [
                    -0.04765648394823074,
                    -0.18347612023353577,
                    0.24941012263298035,
                    0.022468380630016327,
                    -0.19706891477108002,
                ],
                [
                    -0.09828818589448929,
                    -0.18449409306049347,
                    0.2711920738220215,
                    0.044708192348480225,
                    -0.15743865072727203,
                ],
                [
                    -0.13639293611049652,
                    -0.16482298076152802,
                    0.3018418848514557,
                    0.0815257728099823,
                    -0.15574774146080017,
                ],
                [
                    -0.14846578240394592,
                    -0.17294010519981384,
                    0.31513816118240356,
                    0.10425455123186111,
                    -0.16388092935085297,
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
                    -1.3690122365951538,
                    -2.0469894409179688,
                    2.1774630546569824,
                    -0.35028770565986633,
                    -1.7614551782608032,
                ],
                [
                    -0.892522394657135,
                    -1.3144632577896118,
                    2.683281898498535,
                    -1.4629074335098267,
                    -3.3516180515289307,
                ],
                [
                    -1.3936796188354492,
                    0.21844607591629028,
                    4.501010417938232,
                    -0.15485064685344696,
                    -2.651848316192627,
                ],
                [
                    -1.7388781309127808,
                    -0.7211084365844727,
                    3.463726043701172,
                    -0.2992384433746338,
                    -2.65508770942688,
                ],
            ]
        ),
    )


def test_training_step_and_on_epoch_end(batch, model, config):
    assert model._get_metric(TRAINING) is None
    loss = model.training_step(batch, batch_idx=0)
    assert loss is not None
    torch.testing.assert_close(loss, torch.tensor(1.730902075767517))

    model.on_train_epoch_end()


def test_validation_step_and_on_epoch_end(batch, model, config):
    metric = model._get_metric(VALIDATION)
    metric.reset()
    loss = model.validation_step(batch, batch_idx=0)
    assert loss is not None
    torch.testing.assert_close(loss, torch.tensor(1.730902075767517))
    metric_values = {k: v.item() for k, v in metric.compute().items()}
    assert metric_values == {
        "span/ORG/f1": 0.0,
        "span/ORG/precision": 0.0,
        "span/ORG/recall": 0.0,
        "span/PER/f1": 0.0,
        "span/PER/precision": 0.0,
        "span/PER/recall": 0.0,
        "span/macro/f1": 0.0,
        "span/macro/precision": 0.0,
        "span/macro/recall": 0.0,
        "span/micro/f1": 0.0,
        "span/micro/precision": 0.0,
        "span/micro/recall": 0.0,
        "token/macro/f1": 0.0,
        "token/micro/f1": 0.0,
        "token/macro/precision": 0.0,
        "token/macro/recall": 0.0,
        "token/micro/precision": 0.0,
        "token/micro/recall": 0.0,
    }

    model.on_validation_epoch_end()


def test_test_step_and_on_epoch_end(batch, model, config):
    metric = model._get_metric(TESTING)
    metric.reset()
    loss = model.test_step(batch, batch_idx=0)
    assert loss is not None
    torch.testing.assert_close(loss, torch.tensor(1.730902075767517))
    metric_values = {k: v.item() for k, v in metric.compute().items()}
    assert metric_values == {
        "span/ORG/f1": 0.0,
        "span/ORG/precision": 0.0,
        "span/ORG/recall": 0.0,
        "span/PER/f1": 0.0,
        "span/PER/precision": 0.0,
        "span/PER/recall": 0.0,
        "span/macro/f1": 0.0,
        "span/macro/precision": 0.0,
        "span/macro/recall": 0.0,
        "span/micro/f1": 0.0,
        "span/micro/precision": 0.0,
        "span/micro/recall": 0.0,
        "token/macro/f1": 0.0,
        "token/micro/f1": 0.0,
        "token/macro/precision": 0.0,
        "token/macro/recall": 0.0,
        "token/micro/precision": 0.0,
        "token/micro/recall": 0.0,
    }

    model.on_test_epoch_end()


@pytest.mark.parametrize("test_step", [False, True])
def test_predict_and_predict_step(model, batch, config, test_step):
    torch.manual_seed(42)
    if test_step:
        predictions = model.predict_step(batch, batch_idx=0, dataloader_idx=0)
    else:
        predictions = model.predict(batch[0])
    assert set(predictions) == {"labels", "probabilities"}

    assert predictions["labels"].shape == batch[1]["labels"].shape
    torch.testing.assert_close(
        predictions["labels"],
        torch.tensor(
            [
                [-100, 2, 3, 2, 2, -100, -100, -100, -100, -100, -100, -100],
                [-100, 2, 2, 2, 2, 2, 2, 2, 2, -100, -100, -100],
                [-100, 2, 2, 2, 2, 2, 2, 2, -100, -100, -100, -100],
                [-100, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, -100],
            ]
        ),
    )
    torch.testing.assert_close(
        # just check the first two batch entries
        predictions["probabilities"][:2].round(decimals=4),
        torch.tensor(
            [
                [
                    [0.1792, 0.1912, 0.2443, 0.1613, 0.2240],
                    [0.2083, 0.1750, 0.2395, 0.2030, 0.1742],
                    [0.1934, 0.1837, 0.2047, 0.2381, 0.1801],
                    [0.2034, 0.1779, 0.2285, 0.1917, 0.1986],
                    [0.1752, 0.1725, 0.2618, 0.2065, 0.1840],
                    [0.1805, 0.1870, 0.2365, 0.2025, 0.1935],
                    [0.1844, 0.1781, 0.2738, 0.1936, 0.1700],
                    [0.1958, 0.1784, 0.2612, 0.1937, 0.1711],
                    [0.1941, 0.1694, 0.2612, 0.2082, 0.1671],
                    [0.1831, 0.1680, 0.2650, 0.2113, 0.1726],
                    [0.1740, 0.1691, 0.2697, 0.2164, 0.1707],
                    [0.1713, 0.1672, 0.2723, 0.2205, 0.1687],
                ],
                [
                    [0.1654, 0.1989, 0.2729, 0.1542, 0.2086],
                    [0.1787, 0.1511, 0.3093, 0.1968, 0.1641],
                    [0.1888, 0.1966, 0.2365, 0.2081, 0.1700],
                    [0.2092, 0.1935, 0.2428, 0.2034, 0.1511],
                    [0.2275, 0.1784, 0.2546, 0.1856, 0.1539],
                    [0.2254, 0.1959, 0.2377, 0.1873, 0.1536],
                    [0.2177, 0.1879, 0.2485, 0.1975, 0.1484],
                    [0.2227, 0.1906, 0.2541, 0.1906, 0.1420],
                    [0.2080, 0.2098, 0.2667, 0.1764, 0.1391],
                    [0.1815, 0.2015, 0.2600, 0.1852, 0.1718],
                    [0.1672, 0.1883, 0.3065, 0.1773, 0.1607],
                    [0.1750, 0.1846, 0.2911, 0.1862, 0.1630],
                ],
            ]
        ),
    )


def test_configure_optimizers(model):
    optimizer = model.configure_optimizers()
    assert optimizer is not None
    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.defaults["lr"] == 1e-05
    assert len(optimizer.param_groups) == 1
    assert len(optimizer.param_groups[0]["params"]) > 0
    assert set(optimizer.param_groups[0]["params"]) == set(model.parameters())
