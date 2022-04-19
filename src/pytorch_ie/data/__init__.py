from typing import Any, Dict

from pytorch_ie.data.annotations import (
    Annotation,
    AnnotationList,
    BinaryRelation,
    Label,
    LabeledSpan,
    MultiLabel,
    Span,
)
from pytorch_ie.data.document import Document, TextDocument, annotation_field

Metadata = Dict[str, Any]

__all__ = [
    "Document",
    "TextDocument",
    "Annotation",
    "AnnotationList",
    "Span",
    "LabeledSpan",
    "Label",
    "MultiLabel",
    "BinaryRelation",
    # utility
    "Metadata",
    "annotation_field",
]
