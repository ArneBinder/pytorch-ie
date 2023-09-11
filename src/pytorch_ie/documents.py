import dataclasses
from typing import Any, Dict, Optional, Tuple

from typing_extensions import TypeAlias

from pytorch_ie.annotations import BinaryRelation, Label, LabeledSpan, MultiLabel, Span
from pytorch_ie.core import AnnotationList, Document, annotation_field


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
    pass


# backwards compatibility
TextDocument: TypeAlias = TextBasedDocument


@dataclasses.dataclass
class DocumentWithLabel(Document):
    label: AnnotationList[Label] = annotation_field()


@dataclasses.dataclass
class DocumentWithMultiLabel(Document):
    label: AnnotationList[MultiLabel] = annotation_field()


@dataclasses.dataclass
class TextDocumentWithLabel(DocumentWithLabel, TextBasedDocument):
    pass


@dataclasses.dataclass
class TextDocumentWithMultiLabel(DocumentWithMultiLabel, TextBasedDocument):
    pass


@dataclasses.dataclass
class TextDocumentWithLabeledPartitions(TextBasedDocument):
    partitions: AnnotationList[LabeledSpan] = annotation_field(target="text")


@dataclasses.dataclass
class TextDocumentWithSentences(TextBasedDocument):
    sentences: AnnotationList[Span] = annotation_field(target="text")


@dataclasses.dataclass
class TextDocumentWithEntities(TextBasedDocument):
    entities: AnnotationList[Span] = annotation_field(target="text")


@dataclasses.dataclass
class TextDocumentWithLabeledEntities(TextBasedDocument):
    entities: AnnotationList[LabeledSpan] = annotation_field(target="text")


@dataclasses.dataclass
class TextDocumentWithLabeledEntitiesAndLabeledPartitions(
    TextDocumentWithLabeledEntities, TextDocumentWithLabeledPartitions
):
    pass


@dataclasses.dataclass
class TextDocumentWithLabeledEntitiesAndSentences(
    TextDocumentWithLabeledEntities, TextDocumentWithSentences
):
    pass


@dataclasses.dataclass
class TextDocumentWithLabeledEntitiesAndRelations(TextDocumentWithLabeledEntities):
    relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")


@dataclasses.dataclass
class TextDocumentWithLabeledEntitiesRelationsAndLabeledPartitions(
    TextDocumentWithLabeledEntitiesAndRelations, TextDocumentWithLabeledPartitions
):
    pass


@dataclasses.dataclass
class TextDocumentWithEntitiesAndRelations(TextDocumentWithEntities):
    relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")


@dataclasses.dataclass
class TextDocumentWithEntitiesRelationsAndLabeledPartitions(
    TextDocumentWithEntitiesAndRelations, TextDocumentWithLabeledPartitions
):
    pass
