from typing import Any, Dict, Generic, Optional, Sequence, Union

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from pytorch_ie.core.taskmodule import (
    DocumentType,
    InputEncoding,
    IterableTaskEncodingDataset,
    TargetEncoding,
    TaskEncoding,
    TaskEncodingDataset,
    TaskModule,
)


class DataModule(LightningDataModule, Generic[DocumentType, InputEncoding, TargetEncoding]):
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
        taskmodule: TaskModule[DocumentType, InputEncoding, TargetEncoding, Any, Any, Any],
        dataset: Dict[str, Sequence[DocumentType]],
        data_config_path: Optional[str] = None,
        train_split: Optional[str] = "train",
        val_split: Optional[str] = "validation",
        test_split: Optional[str] = "test",
        prepare_split: Optional[str] = None,
        **dataloader_kwargs,
    ):
        super().__init__()

        self.taskmodule = taskmodule
        self.config_path = data_config_path
        self.dataset = dataset
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        # per default, use train data to prepare the taskmodule
        self.prepare_split = prepare_split or self.train_split
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

    @property
    def num_train(self) -> int:
        if self.train_split is None:
            raise ValueError("no train_split assigned")
        data_train = self._data.get(self.train_split, None)
        if data_train is None:
            raise ValueError("can not get train size if setup() was not yet called")
        if isinstance(data_train, IterableTaskEncodingDataset):
            raise TypeError("IterableTaskEncodingDataset has no length")
        return len(data_train)

    def setup(self, stage: Optional[str] = None, **kwargs):

        if stage == "fit" or stage is None:
            if self.prepare_split is None:
                raise ValueError(f"prepare_data_split is required to prepare the taskmodule")
            self.taskmodule.prepare(self.dataset[self.prepare_split])

        for split in [self.train_split, self.val_split, self.test_split]:
            if split is None or split not in self.dataset:
                continue
            task_encoding_dataset = self.taskmodule.encode(
                self.dataset[split], encode_target=True, as_dataset=True
            )
            if not isinstance(
                task_encoding_dataset, (TaskEncodingDataset, IterableTaskEncodingDataset)
            ):
                raise TypeError(
                    f"taskmodule.encode did not return a (Iterable)TaskEncodingDataset, but: {type(task_encoding_dataset)}"
                )
            self._data[split] = task_encoding_dataset

    def data_split(
        self, split: Optional[str] = None
    ) -> Union[
        TaskEncodingDataset[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]],
        IterableTaskEncodingDataset[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]],
    ]:
        if split is None or split not in self._data:
            raise ValueError(f"data for split={split} not available")
        return self._data[split]

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_split(self.train_split),
            collate_fn=self.taskmodule.collate,
            shuffle=True,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_split(self.val_split),
            collate_fn=self.taskmodule.collate,
            shuffle=False,
            **self.dataloader_kwargs,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_split(self.test_split),
            collate_fn=self.taskmodule.collate,
            shuffle=False,
            **self.dataloader_kwargs,
        )
