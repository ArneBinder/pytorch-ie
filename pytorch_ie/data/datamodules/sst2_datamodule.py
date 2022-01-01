from typing import List, Optional, Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from pytorch_ie.data.datasets.sst2 import load_sst2
from pytorch_ie.taskmodules.taskmodule import TaskEncoding, TaskModule


class SST2DataModule(LightningDataModule):
    """
    Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        task_module: TaskModule,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.task_module = task_module
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.data_train: Optional[List[TaskEncoding]] = None
        self.data_val: Optional[List[TaskEncoding]] = None

    @property
    def num_train(self) -> int:
        return self.train_val_split[0]

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        train_documents = load_sst2(split="train")
        val_documents = load_sst2(split="validation")

        self.task_module.prepare(train_documents)

        self.data_train = self.task_module.encode(train_documents, encode_target=True)
        self.data_val = self.task_module.encode(val_documents, encode_target=True)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.task_module.collate,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.task_module.collate,
            shuffle=False,
        )
