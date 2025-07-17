# flake8: noqa

from pytorch_ie.auto import AutoModel, AutoPipeline, AutoTaskModule
from pytorch_ie.core import *
from pytorch_ie.datamodule import PieDataModule
from pytorch_ie.dataset import IterableTaskEncodingDataset, TaskEncodingDataset
from pytorch_ie.pipeline import PyTorchIEPipeline

# kept for backward compatibility
Pipeline = PyTorchIEPipeline
