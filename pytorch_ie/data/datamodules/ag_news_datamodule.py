from typing import List, Optional, Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from pytorch_ie.data.datasets.ag_news import load_ag_news
from pytorch_ie.taskmodules.taskmodule import TaskEncoding, TaskModule


class AgNewsDataModule(LightningDataModule):
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
        train_val_split: Tuple[int, int] = (108000, 12000),
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.task_module = task_module
        self.train_val_split = train_val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.data_train: Optional[List[TaskEncoding]] = None
        self.data_val: Optional[List[TaskEncoding]] = None
        self.data_test: Optional[List[TaskEncoding]] = None

    @property
    def num_train(self) -> int:
        return self.train_val_split[0]

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        train_documents = load_ag_news(split="train")
        test_documents = load_ag_news(split="test")

        self.task_module.prepare(train_documents)

        train_set = self.task_module.encode(train_documents, encode_target=True)
        test_set = self.task_module.encode(test_documents, encode_target=True)

        # TODO: fix mypy: Incompatible types in assignment (expression has type "Subset[<nothing>]", variable has type "Optional[List[TaskEncoding[Any, Any]]]")
        self.data_train, self.data_val = random_split(train_set, self.train_val_split)
        self.data_test = test_set

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

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.task_module.collate,
            shuffle=False,
        )
