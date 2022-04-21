from typing import Any, Dict

from .builder import GeneratorBasedBuilder
from .dataset import Dataset
from .dataset_formatter import DocumentFormatter

Metadata = Dict[str, Any]

__all__ = [
    "Metadata",
    "GeneratorBasedBuilder",
    "Dataset",
    "DocumentFormatter",
]
