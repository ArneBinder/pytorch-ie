from typing import Any, Dict

from pytorch_ie.data.annotations import (
    Annotation,
    AnnotationList,
    BinaryRelation,
    Label,
    LabeledMultiSpan,
    LabeledSpan,
    MultiLabel,
    MultiLabeledBinaryRelation,
    MultiLabeledMultiSpan,
    MultiLabeledSpan,
    Span,
)
from pytorch_ie.data.document import Document, TextDocument, annotation_field

Metadata = Dict[str, Any]

__all__ = [
    "Document",
    "TextDocument",
    "Annotation",
    "AnnotationList",
    "BinaryRelation",
    "MultiLabeledBinaryRelation",
    "Label",
    "MultiLabel",
    "Span",
    "LabeledSpan",
    "MultiLabeledSpan",
    "LabeledMultiSpan",
    "MultiLabeledMultiSpan",
    "Metadata",
    "annotation_field",
]
