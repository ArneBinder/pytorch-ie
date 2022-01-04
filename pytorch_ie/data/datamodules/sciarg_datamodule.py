from typing import Optional

from pytorch_ie.data.datamodules.datamodule import DataModule
from pytorch_ie.data.annotation_utils import annotate_dataset_with_sections
from pytorch_ie.data.datasets.brat import load_brat


class SciargDataModule(DataModule):

    def __init__(self, **kwargs):
        super().__init__(
            data_config_path='./pytorch_ie/data/datasets/hf_datasets/sciarg.json',
            load_data=load_brat,
            dataset_preprocessing_hook=annotate_dataset_with_sections,
            **kwargs
        )

    def setup(
        self,
        stage: Optional[str] = None,
        **kwargs
    ):
        # for now, hardcode train_test_split
        super().setup(stage=stage,  train_test_split={"train_size": 30}, **kwargs)
