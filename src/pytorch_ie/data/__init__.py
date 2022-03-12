from typing import Any, Dict, List

from pytorch_ie.data.document import BinaryRelation, Document, LabeledSpan

Metadata = Dict[str, Any]
DatasetDict = Dict[str, List[Document]]

__all__ = [
    # TODO: this should also directly export "DataModule",
    #  but this causes a cyclic import for now
    "DatasetDict",
    "Document",
    "LabeledSpan",
    "BinaryRelation",
    # utility
    "Metadata",
]
