import json
from typing import List, Optional, Callable

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from pytorch_ie.taskmodules.taskmodule import TaskEncoding, TaskModule


class DataModule(LightningDataModule):
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
        load_data: Callable,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        data_config_path: Optional[str] = None,
        dataset_preprocessing_hook: Optional[Callable] = None,
    ):
        super().__init__()

        self.task_module = task_module
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.config_path = data_config_path
        self.load_data = load_data
        self.dataset_preprocessing_hook = dataset_preprocessing_hook

        self.data_train: Optional[List[TaskEncoding]] = None
        self.data_val: Optional[List[TaskEncoding]] = None
        self.data_test: Optional[List[TaskEncoding]] = None

    @property
    def num_train(self) -> int:
        return self.train_val_split[0]

    def setup(
        self,
        stage: Optional[str] = None,
        **kwargs
    ):
        if self.config_path is not None:
            load_kwargs = json.load(open(self.config_path))
            load_kwargs.update(kwargs)
        else:
            load_kwargs = kwargs
        all_documents = self.load_data(**load_kwargs)

        for split, data in all_documents.items():
            if self.dataset_preprocessing_hook is not None:
                preprocessed_data = self.dataset_preprocessing_hook(data)
                # dataset_preprocessing_hook might have modified the data inplace and returned None in this case
                if preprocessed_data is not None:
                    data = preprocessed_data
            if split == "train":
                self.task_module.prepare(data)
                self.data_train = self.task_module.encode(data, encode_target=True)
            elif split == "val":
                self.data_val = self.task_module.encode(data, encode_target=True)
            elif split == "test":
                self.data_test = self.task_module.encode(data, encode_target=True)
            else:
                raise ValueError(f'Unknowns split identifier: "{split}". Use one of "train", "val", or "test".')

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
