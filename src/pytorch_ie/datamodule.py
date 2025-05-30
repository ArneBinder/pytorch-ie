from typing import Any, Dict, Generic, Iterator, Optional, Sequence, TypeVar, Union

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from pytorch_ie.core import Document
from pytorch_ie.core.taskmodule import (
    IterableTaskEncodingDataset,
    TaskEncoding,
    TaskEncodingDataset,
    TaskModule,
)

DocumentType = TypeVar("DocumentType", bound=Document)
InputEncoding = TypeVar("InputEncoding")
TargetEncoding = TypeVar("TargetEncoding")


class PieDataModule(LightningDataModule, Generic[DocumentType, InputEncoding, TargetEncoding]):
    """A simple LightningDataModule for PIE document datasets.

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
        taskmodule: TaskModule[DocumentType, InputEncoding, TargetEncoding, Any, Any, Any],
        dataset: Dict[str, Sequence[DocumentType]],
        data_config_path: Optional[str] = None,
        train_split: Optional[str] = "train",
        val_split: Optional[str] = "validation",
        test_split: Optional[str] = "test",
        show_progress_for_encode: bool = False,
        **dataloader_kwargs,
    ):
        super().__init__()

        self.taskmodule = taskmodule
        self.config_path = data_config_path
        self.dataset = dataset
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.show_progress_for_encode = show_progress_for_encode
        self.dataloader_kwargs = dataloader_kwargs

        self._data: Dict[
            str,
            Union[
                TaskEncodingDataset[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]],
                IterableTaskEncodingDataset[
                    TaskEncoding[DocumentType, InputEncoding, TargetEncoding]
                ],
            ],
        ] = {}

    def get_split_size(self, split: str) -> int:
        data = self._data.get(split, None)
        if data is None:
            raise ValueError(f"can not get {split} size if setup() was not yet called")
        if isinstance(data, IterableTaskEncodingDataset):
            raise TypeError("IterableTaskEncodingDataset has no length")
        return len(data)

    @property
    def num_train(self) -> int:
        if self.train_split is None:
            raise ValueError("no train split assigned")
        return self.get_split_size(self.train_split)

    @property
    def num_val(self) -> int:
        if self.val_split is None:
            raise ValueError("no val split assigned")
        return self.get_split_size(self.val_split)

    @property
    def num_test(self) -> int:
        if self.test_split is None:
            raise ValueError("no test split assigned")
        return self.get_split_size(self.test_split)

    def setup(self, stage: str) -> None:

        if stage == "fit":
            split_names = [self.train_split, self.val_split]
        elif stage == "validate":
            split_names = [self.val_split]
        elif stage == "test":
            split_names = [self.test_split]
        else:
            raise NotImplementedError(f"not implemented for stage={stage} ")

        for split in split_names:
            if split is None or split not in self.dataset:
                continue
            task_encodings = self.taskmodule.encode(
                self.dataset[split],
                encode_target=True,
                show_progress=self.show_progress_for_encode,
            )
            if isinstance(task_encodings, Sequence):
                self._data[split] = TaskEncodingDataset(task_encodings)
            elif isinstance(task_encodings, Iterator):
                self._data[split] = IterableTaskEncodingDataset(task_encodings)
            else:
                raise TypeError(
                    f"task_encodings should be a Sequence or Iterator, but got {type(task_encodings)}"
                )

    def data_split(self, split: Optional[str] = None) -> Union[
        TaskEncodingDataset[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]],
        IterableTaskEncodingDataset[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]],
    ]:
        if split is None or split not in self._data:
            raise ValueError(f"data for split={split} not available")
        return self._data[split]

    def train_dataloader(
        self,
    ) -> DataLoader[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]]:
        ds = self.data_split(self.train_split)
        return DataLoader(
            dataset=ds,
            collate_fn=self.taskmodule.collate,
            # don't shuffle streamed datasets
            shuffle=not isinstance(ds, IterableTaskEncodingDataset),
            **self.dataloader_kwargs,
        )

    def val_dataloader(
        self,
    ) -> DataLoader[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]]:
        return DataLoader(
            dataset=self.data_split(self.val_split),
            collate_fn=self.taskmodule.collate,
            shuffle=False,
            **self.dataloader_kwargs,
        )

    def test_dataloader(
        self,
    ) -> DataLoader[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]]:
        return DataLoader(
            dataset=self.data_split(self.test_split),
            collate_fn=self.taskmodule.collate,
            shuffle=False,
            **self.dataloader_kwargs,
        )
