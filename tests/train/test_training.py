from typing import Iterator, List

import pytest
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from pytorch_ie.core import Document, PyTorchIEModel, TaskModule
from pytorch_ie.data.datamodules.datamodule import IterableTaskEncodingDataset
from pytorch_ie.models import TransformerTokenClassificationModel
from pytorch_ie.taskmodules import TransformerTokenClassificationTaskModule

MODEL_NAME = "bert-base-cased"


@pytest.fixture
def prepared_taskmodule(documents):
    taskmodule = TransformerTokenClassificationTaskModule(
        tokenizer_name_or_path=MODEL_NAME,
        max_length=128,
    )
    taskmodule.prepare(documents)
    return taskmodule


@pytest.fixture
def model(prepared_taskmodule):
    model = TransformerTokenClassificationModel(
        model_name_or_path=MODEL_NAME,
        num_classes=len(prepared_taskmodule.label_to_id),
        learning_rate=1e-4,
    )
    return model


@pytest.mark.parametrize("encode_batch_size", [None, 2])
def test_transformer_token_classification(
    model, prepared_taskmodule, documents, encode_batch_size
):
    pl.seed_everything(42)

    num_epochs = 1
    batch_size = 32

    train_dataset = prepared_taskmodule.encode(
        documents, encode_target=True, batch_size=encode_batch_size
    )

    if isinstance(train_dataset, Iterator):
        train_dataset = IterableTaskEncodingDataset(train_dataset)
        shuffle = False
    else:
        shuffle = True
    train_dataloader = DataLoader(
        train_dataset,  # type: ignore
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=prepared_taskmodule.collate,
    )

    trainer = pl.Trainer(
        fast_dev_run=False,
        max_epochs=num_epochs,
        gpus=0,
        checkpoint_callback=False,
        precision=32,
    )
    trainer.fit(model, train_dataloader)
