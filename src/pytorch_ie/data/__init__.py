from typing import Any, Dict, List

from pytorch_ie.data.document import BinaryRelation, Document, LabeledSpan

Metadata = Dict[str, Any]
DatasetDict = Dict[str, List[Document]]

__all__ = [
    "DatasetDict",
    "Document",
    "LabeledSpan",
    "BinaryRelation",
    # utility
    "Metadata",
]
