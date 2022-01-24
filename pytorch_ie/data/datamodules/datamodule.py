import logging
from typing import Any, Dict, Generic, List, Mapping, Optional, Sequence, Tuple, Union

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset

from pytorch_ie import Document
from pytorch_ie.taskmodules.taskmodule import (
    InputEncoding,
    TargetEncoding,
    TaskEncoding,
    TaskModule,
)

logger = logging.getLogger(__name__)


class TaskEncodingDataset(
    Dataset[TaskEncoding[InputEncoding, TargetEncoding]],
    Generic[InputEncoding, TargetEncoding],
):
    def __init__(self, encodings: List[TaskEncoding[InputEncoding, TargetEncoding]]):
        self._encodings = encodings

    def __getitem__(self, index) -> TaskEncoding[InputEncoding, TargetEncoding]:
        return self._encodings[index]

    def __len__(self):
        return len(self._encodings)


class DataModule(LightningDataModule, Generic[InputEncoding, TargetEncoding]):
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
        task_module: TaskModule[InputEncoding, TargetEncoding, Any, Any, Any],
        dataset: Dict[str, List[Document]],
        val_size: Optional[Union[int, float]] = None,
        train_size: Optional[Union[int, float]] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        data_config_path: Optional[str] = None,
        prepare_data_split: str = "train",
        train_split_name: str = "train",
        val_split_name: str = "val",
        test_split_name: str = "test",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.task_module = task_module
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.config_path = data_config_path
        self.dataset = dataset
        self.prepare_data_split = prepare_data_split
        self.val_size = val_size
        self.train_size = train_size
        self.train_split_name = train_split_name
        self.val_split_name = val_split_name
        self.test_split_name = test_split_name

        self.data_train: Optional[TaskEncodingDataset[InputEncoding, TargetEncoding]] = None
        self.data_val: Optional[TaskEncodingDataset[InputEncoding, TargetEncoding]] = None
        self.data_test: Optional[TaskEncodingDataset[InputEncoding, TargetEncoding]] = None

    @property
    def num_train(self) -> int:
        if self.data_train is None:
            raise ValueError("can not get train size if setup() was not yet called")
        return len(self.data_train)

    def setup(self, stage: Optional[str] = None, **kwargs):

        # execute before looping over dataset splits to ensure that this
        # will be executed before eny call to task_module.encode()
        self.task_module.prepare(self.dataset[self.prepare_data_split])

        for split, data in self.dataset.items():

            if split == self.train_split_name:
                self.data_train = TaskEncodingDataset(
                    self.task_module.encode(data, encode_target=True)
                )
            elif split == self.val_split_name:
                self.data_val = TaskEncodingDataset(
                    self.task_module.encode(data, encode_target=True)
                )
            elif split == self.test_split_name:
                self.data_test = TaskEncodingDataset(
                    self.task_module.encode(data, encode_target=True)
                )
            else:
                raise ValueError(
                    f'Unknowns split identifier: "{split}". Use one of "train", "val", or "test".'
                )

        if self.val_size is not None:
            assert (
                self.data_train is not None
            ), "data_train has to be set to create random train dev splits from it"

            if isinstance(self.val_size, float):
                val_size = int(self.val_size * len(self.data_train))
            elif isinstance(self.val_size, int):
                val_size = self.val_size
            else:
                raise ValueError(
                    f"'val_size' has unknown type={type(self.val_size)}. Should be either int or float."
                )

            _train_size = self.train_size
            if _train_size is None:
                _train_size = len(self.data_train) - val_size

            if isinstance(_train_size, float):
                train_size = int(_train_size * len(self.data_train))
            elif isinstance(_train_size, int):
                train_size = _train_size
            else:
                raise ValueError(
                    f"'train_size' has unknown type={type(_train_size)}. Should be either int or float."
                )

            sizes_total = val_size + train_size
            data_to_split: TaskEncodingDataset[InputEncoding, TargetEncoding]
            if len(self.data_train) > sizes_total:
                logger.warning(
                    f"the created splits do not take all available data into account: only {sizes_total} will be "
                    f"used in new train/val splits, but original train data contains {len(self.data_train)} entries"
                )
                # TODO: check mypy error that says that the type of the expression is:
                #  "TaskEncoding[InputEncoding, TargetEncoding]", but should be
                #  "TaskEncodingDataset[InputEncoding, TargetEncoding]"
                data_to_split = self.data_train[:sizes_total]  # type: ignore
            else:
                data_to_split = self.data_train
            # type checking is broken for random_split, so we ignore it
            self.data_train, self.data_val = random_split(data_to_split, [train_size, val_size])  # type: ignore

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
