from typing import Any, Dict, List

from pytorch_ie.data.datamodules.datamodule import DataModule
from pytorch_ie.data.document import BinaryRelation, Document, LabeledSpan

Metadata = Dict[str, Any]
DatasetDict = Dict[str, List[Document]]

__all__ = [
    "DataModule",
    "DatasetDict",
    "Document",
    "LabeledSpan",
    "BinaryRelation",
    # utility
    "Metadata",
]
