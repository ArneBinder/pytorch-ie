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
        random_train_val_split: Optional[
            Union[Dict[str, Union[float, int]], Tuple[Union[float, int], Union[float, int]]]
        ] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        data_config_path: Optional[str] = None,
        prepare_data_split: str = "train",
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
        self.random_train_val_split = random_train_val_split

        self.data_train: Optional[TaskEncodingDataset[InputEncoding, TargetEncoding]] = None
        self.data_val: Optional[TaskEncodingDataset[InputEncoding, TargetEncoding]] = None
        self.data_test: Optional[TaskEncodingDataset[InputEncoding, TargetEncoding]] = None

    @property
    def num_train(self) -> int:
        if self.data_train is None:
            raise ValueError("can not get train size if setup() was not yet called")
        return len(self.data_train)

    def setup(self, stage: Optional[str] = None, **kwargs):

        for split, data in self.dataset.items():

            if split == self.prepare_data_split:
                self.task_module.prepare(data)

            if split == "train":
                self.data_train = TaskEncodingDataset(
                    self.task_module.encode(data, encode_target=True)
                )
            elif split == "val":
                self.data_val = TaskEncodingDataset(
                    self.task_module.encode(data, encode_target=True)
                )
            elif split == "test":
                self.data_test = TaskEncodingDataset(
                    self.task_module.encode(data, encode_target=True)
                )
            else:
                raise ValueError(
                    f'Unknowns split identifier: "{split}". Use one of "train", "val", or "test".'
                )

        if self.random_train_val_split is not None:
            assert (
                self.data_train is not None
            ), "data_train has to be set to create random train dev splits from it"

            # if random_train_val_split is a sequence, we use its values for [train, val] sizes
            sizes: Dict[str, Union[int, float]]
            if isinstance(self.random_train_val_split, Sequence):
                assert (
                    len(self.random_train_val_split) == 2
                ), "if not all split sizes are specified, random_train_val_split has to be a dict"
                sizes = dict(zip(["train", "val"], self.random_train_val_split))
            elif isinstance(self.random_train_val_split, Mapping):
                sizes = self.random_train_val_split
            else:
                raise ValueError(
                    f"split length specifiers has unknown type={type(self.random_train_val_split)}: "
                    f"{self.random_train_val_split}"
                )
            # convert percentages to absolute, if necessary
            num_documents: Dict[str, int] = {}
            for split in sizes:
                s = sizes[split]
                if isinstance(s, int):
                    num_documents[split] = s
                elif isinstance(s, float):
                    assert 0.0 <= s <= 1.0, (
                        f"if split size is specified as percentage (i.e as float value), it has to be between "
                        f"0.0 and 1.0"
                    )
                    num_documents[split] = int(s * len(self.data_train))
                else:
                    raise ValueError(
                        f"split length specifier has unknown type={type(sizes[split])}: "
                        f"{sizes[split]}"
                    )
            # set missing sizes, if not specified, for train or val set
            for missing, other in [("train", "val"), ("val", "train")]:
                if missing not in num_documents:
                    assert (
                        other in num_documents
                    ), f"if no {missing} split size is specified, a {other} split size has to be provided"
                    num_documents[missing] = len(self.data_train) - num_documents[other]

            logger.info(f"split train data randomly into new sets: {num_documents}")
            # type checking is broken for random_split, so we ignore it
            self.data_train, self.data_val = random_split(  # type: ignore
                self.data_train, [num_documents[split] for split in ["train", "val"]]
            )

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
