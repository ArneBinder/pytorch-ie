from typing import List

import pytest
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from pytorch_ie import Document
from pytorch_ie.core.pytorch_ie import PyTorchIEModel
from pytorch_ie.models import TransformerTokenClassificationModel
from pytorch_ie.taskmodules import TransformerTokenClassificationTaskModule
from pytorch_ie.taskmodules.taskmodule import TaskModule


def _test_training(model: PyTorchIEModel, task_module: TaskModule, documents: List[Document]):
    pl.seed_everything(42)

    num_epochs = 1
    batch_size = 32

    train_dataset = task_module.encode(documents, encode_target=True)

    train_dataloader = DataLoader(
        train_dataset,  # type: ignore
        batch_size=batch_size,
        shuffle=True,
        collate_fn=task_module.collate,
    )

    trainer = pl.Trainer(
        fast_dev_run=False,
        max_epochs=num_epochs,
        gpus=0,
        checkpoint_callback=False,
        precision=32,
    )
    trainer.fit(model, train_dataloader)


@pytest.mark.slow
def test_transformer_token_classification(documents):
    model_name = "bert-base-cased"

    task_module = TransformerTokenClassificationTaskModule(
        tokenizer_name_or_path=model_name,
        max_length=128,
    )

    task_module.prepare(documents)
    model = TransformerTokenClassificationModel(
        model_name_or_path=model_name,
        num_classes=len(task_module.label_to_id),
        learning_rate=1e-4,
    )

    _test_training(model=model, task_module=task_module, documents=documents)
