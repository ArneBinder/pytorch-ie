import dataclasses
from typing import Any, Dict, Optional, Tuple

from typing_extensions import TypeAlias

from pytorch_ie.annotations import BinaryRelation, Label, LabeledSpan, MultiLabel, Span
from pytorch_ie.core import AnnotationLayer, Document, annotation_field


@dataclasses.dataclass
class WithMetadata:
    id: Optional[str] = None
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class WithTokens:
    tokens: Tuple[str, ...]


@dataclasses.dataclass
class WithText:
    text: str


@dataclasses.dataclass
class TextBasedDocument(WithMetadata, WithText, Document):
    pass


@dataclasses.dataclass
class TokenBasedDocument(WithMetadata, WithTokens, Document):
    def __post_init__(self) -> None:

        # When used in a dataset, the document gets serialized to json like structure which does not know tuples,
        # so they get converted to lists. This is a workaround to automatically convert the "tokens" back to tuples
        # when the document is created from a dataset.
        if isinstance(self.tokens, list):
            object.__setattr__(self, "tokens", tuple(self.tokens))
        elif not isinstance(self.tokens, tuple):
            raise ValueError("tokens must be a tuple.")

        # Call the default document construction code
        super().__post_init__()


# backwards compatibility
TextDocument: TypeAlias = TextBasedDocument


@dataclasses.dataclass
class DocumentWithLabel(Document):
    label: AnnotationLayer[Label] = annotation_field()


@dataclasses.dataclass
class DocumentWithMultiLabel(Document):
    label: AnnotationLayer[MultiLabel] = annotation_field()


@dataclasses.dataclass
class TextDocumentWithLabel(DocumentWithLabel, TextBasedDocument):
    pass


@dataclasses.dataclass
class TextDocumentWithMultiLabel(DocumentWithMultiLabel, TextBasedDocument):
    pass


@dataclasses.dataclass
class TextDocumentWithLabeledPartitions(TextBasedDocument):
    labeled_partitions: AnnotationLayer[LabeledSpan] = annotation_field(target="text")


@dataclasses.dataclass
class TextDocumentWithSentences(TextBasedDocument):
    sentences: AnnotationLayer[Span] = annotation_field(target="text")


@dataclasses.dataclass
class TextDocumentWithSpans(TextBasedDocument):
    spans: AnnotationLayer[Span] = annotation_field(target="text")


@dataclasses.dataclass
class TextDocumentWithLabeledSpans(TextBasedDocument):
    labeled_spans: AnnotationLayer[LabeledSpan] = annotation_field(target="text")


@dataclasses.dataclass
class TextDocumentWithLabeledSpansAndLabeledPartitions(
    TextDocumentWithLabeledSpans, TextDocumentWithLabeledPartitions
):
    pass


@dataclasses.dataclass
class TextDocumentWithLabeledSpansAndSentences(
    TextDocumentWithLabeledSpans, TextDocumentWithSentences
):
    pass


@dataclasses.dataclass
class TextDocumentWithLabeledSpansAndBinaryRelations(TextDocumentWithLabeledSpans):
    binary_relations: AnnotationLayer[BinaryRelation] = annotation_field(target="labeled_spans")


@dataclasses.dataclass
class TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions(
    TextDocumentWithLabeledSpansAndLabeledPartitions,
    TextDocumentWithLabeledSpansAndBinaryRelations,
    TextDocumentWithLabeledPartitions,
):
    pass


@dataclasses.dataclass
class TextDocumentWithSpansAndBinaryRelations(TextDocumentWithSpans):
    binary_relations: AnnotationLayer[BinaryRelation] = annotation_field(target="spans")


@dataclasses.dataclass
class TextDocumentWithSpansAndLabeledPartitions(
    TextDocumentWithSpans, TextDocumentWithLabeledPartitions
):
    pass


@dataclasses.dataclass
class TextDocumentWithSpansBinaryRelationsAndLabeledPartitions(
    TextDocumentWithSpansAndLabeledPartitions,
    TextDocumentWithSpansAndBinaryRelations,
    TextDocumentWithLabeledPartitions,
):
    pass
