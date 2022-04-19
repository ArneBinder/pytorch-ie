from typing import Any, Dict

from pytorch_ie.data.annotations import (
    Annotation,
    BinaryRelation,
    Label,
    LabeledSpan,
    MultiLabel,
    Span,
)
from pytorch_ie.data.document import Document, TextDocument

Metadata = Dict[str, Any]

__all__ = [
    "Document",
    "TextDocument",
    "Annotation",
    "Span",
    "LabeledSpan",
    "Label",
    "MultiLabel",
    "BinaryRelation",
    # utility
    "Metadata",
]
