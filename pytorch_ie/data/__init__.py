from typing import Any, Dict

from pytorch_ie.data.document import BinaryRelation, Document, LabeledSpan

Metadata = Dict[str, Any]

__all__ = [
    "Document",
    "LabeledSpan",
    "BinaryRelation",
    # utility
    "Metadata",
]
