from abc import ABC, abstractmethod
from typing import Generic, Iterable, Iterator, Optional, Sequence, Type, TypeVar, Union

from pie_core import Document, TaskEncoding, TaskEncodingSequence, TaskModule
from typing_extensions import TypeAlias

DocumentType = TypeVar("DocumentType", bound=Document)
ConvertedDocumentType = TypeVar("ConvertedDocumentType", bound=Document)
InputEncodingType = TypeVar("InputEncodingType")
TargetEncodingType = TypeVar("TargetEncodingType")
# TaskEncoding: defined below
TaskBatchEncodingType = TypeVar("TaskBatchEncodingType")
# ModelBatchEncoding: defined in models
ModelBatchOutputType = TypeVar("ModelBatchOutputType")
TaskOutputType = TypeVar("TaskOutputType")

TaskEncodingType: TypeAlias = TaskEncoding[
    DocumentType,
    InputEncodingType,
    TargetEncodingType,
]


class TaskModuleWithDocumentConverter(
    TaskModule,
    ABC,
    Generic[
        ConvertedDocumentType,
        DocumentType,
        InputEncodingType,
        TargetEncodingType,
        TaskBatchEncodingType,
        ModelBatchOutputType,
        TaskOutputType,
    ],
):
    @property
    def document_type(self) -> Optional[Type[Document]]:
        if super().document_type is not None:
            raise NotImplementedError(f"please overwrite document_type for {type(self).__name__}")
        else:
            return None

    @abstractmethod
    def _convert_document(self, document: DocumentType) -> ConvertedDocumentType:
        """Convert a document of the taskmodule document type to the expected document type of the
        wrapped taskmodule.

        Args:
            document: the input document

        Returns: the converted document
        """
        pass

    def _prepare(self, documents: Sequence[DocumentType]) -> None:
        # use an iterator for lazy processing
        documents_converted = (self._convert_document(doc) for doc in documents)
        super()._prepare(documents=documents_converted)

    def convert_document(self, document: DocumentType) -> ConvertedDocumentType:
        converted_document = self._convert_document(document)
        if "original_document" in converted_document.metadata:
            raise ValueError(
                f"metadata of converted_document has already and entry 'original_document', "
                f"this is not allowed. Please adjust '{type(self).__name__}._convert_document()' "
                f"to produce documents without that key in metadata."
            )
        converted_document.metadata["original_document"] = document
        return converted_document

    def encode(self, documents: Union[DocumentType, Iterable[DocumentType]], **kwargs) -> Union[
        Sequence[TaskEncodingType],
        TaskEncodingSequence[TaskEncodingType, DocumentType],
        Iterator[TaskEncodingType],
    ]:
        converted_documents: Union[DocumentType, Iterable[DocumentType]]
        if isinstance(documents, Document):
            converted_documents = self.convert_document(documents)
        else:
            converted_documents = [self.convert_document(doc) for doc in documents]
        return super().encode(documents=converted_documents, **kwargs)

    def decode(self, **kwargs) -> Sequence[DocumentType]:
        decoded_documents = super().decode(**kwargs)
        result = []
        for doc in decoded_documents:
            original_document = doc.metadata["original_document"]
            self._integrate_predictions_from_converted_document(
                converted_document=doc, document=original_document
            )
            result.append(original_document)
        return result

    @abstractmethod
    def _integrate_predictions_from_converted_document(
        self,
        document: DocumentType,
        converted_document: ConvertedDocumentType,
    ) -> None:
        """Convert the predictions at the respective layers of the converted_document and add them
        to the original document predictions.

        Args:
            document: document to attach the converted predictions to
            converted_document: the document returned by the wrapped taskmodule, including predictions
        """
        pass
