# backwards compatibility
from pie_documents.documents import (
    DocumentWithLabel,
    DocumentWithMultiLabel,
    TextBasedDocument,
    TextDocumentWithLabel,
    TextDocumentWithLabeledMultiSpans,
    TextDocumentWithLabeledMultiSpansAndBinaryRelations,
    TextDocumentWithLabeledMultiSpansAndLabeledPartitions,
    TextDocumentWithLabeledMultiSpansBinaryRelationsAndLabeledPartitions,
    TextDocumentWithLabeledPartitions,
    TextDocumentWithLabeledSpans,
    TextDocumentWithLabeledSpansAndBinaryRelations,
    TextDocumentWithLabeledSpansAndLabeledPartitions,
    TextDocumentWithLabeledSpansAndSentences,
    TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
    TextDocumentWithMultiLabel,
    TextDocumentWithSentences,
    TextDocumentWithSpans,
    TextDocumentWithSpansAndBinaryRelations,
    TextDocumentWithSpansAndLabeledPartitions,
    TextDocumentWithSpansBinaryRelationsAndLabeledPartitions,
    TokenBasedDocument,
    WithMetadata,
    WithText,
    WithTokens,
)
from typing_extensions import TypeAlias

# backwards compatibility
TextDocument: TypeAlias = TextBasedDocument
