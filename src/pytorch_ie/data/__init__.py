from typing import Dict, Union

from datasets import Split

from .builder import GeneratorBasedBuilder
from .dataset import Dataset
from .dataset_formatter import DocumentFormatter

DatasetDict = Dict[Union[str, Split], Dataset]

__all__ = [
    "GeneratorBasedBuilder",
    "Dataset",
    "DatasetDict",
    "DocumentFormatter",
]
