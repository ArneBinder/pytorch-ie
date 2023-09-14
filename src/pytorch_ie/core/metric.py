from abc import ABC, abstractmethod
from typing import Dict, Generic, Iterable, Optional, Type, TypeVar, Union

from pytorch_ie.core.document import Document

T = TypeVar("T")


class DocumentMetric(ABC, Generic[T]):
    """This defines the interface for a document metric."""

    # The document type that this metric can process. Will be used for auto-conversion, if available.
    # Overwrite this if the metric requires a specific document type, e.g. a TextBasedDocument.
    DOCUMENT_TYPE: Optional[Type[Document]] = None

    def __init__(self):
        self.reset()
        self._current_split: Optional[str] = None

    @property
    def document_type(self) -> Optional[Type[Document]]:
        """The document type that this metric can process. Will be used for auto-conversion, if available.
        Overwrite this if the document type depends on some parameters of the metric."""
        return self.DOCUMENT_TYPE

    @abstractmethod
    def reset(self) -> None:
        """Any reset logic that needs to be performed before the metric is called again."""

    def __call__(
        self,
        document_or_collection: Union[Iterable[Document], Document, Dict[str, Iterable[Document]]],
    ) -> Union[Dict[str, T], T]:
        """This method is called to update the metric with a document or collection of documents.

        If a collection is passed, the metric is also computed and the result is returned. If the
        collection is a dictionary, the metric is computed for each split and the result is
        returned as a dictionary.
        """
        if isinstance(document_or_collection, Document):
            # do not reset here to allow for multiple calls
            self._update(document_or_collection)
            return self.compute(reset=False)
        elif isinstance(document_or_collection, dict):
            result: Dict[str, T] = {}
            for split_name, split in document_or_collection.items():
                self._current_split = split_name
                self.reset()
                split_values: T = self(split)  # type: ignore
                result[split_name] = split_values
                self._current_split = None
            return result
        elif isinstance(document_or_collection, Iterable):
            for doc in document_or_collection:
                if not isinstance(doc, Document):
                    raise TypeError(
                        f"document_or_collection contains an object that is not a document: {type(doc)}"
                    )
                self._update(doc)
            # do not reset here to allow for multiple calls
            return self.compute(reset=False)
        else:
            raise TypeError(
                f"document_or_collection has unknown type: {type(document_or_collection)}"
            )

    def compute(self, reset: bool = True) -> T:
        metric_values = self._compute()
        if reset:
            self.reset()
        return metric_values

    @abstractmethod
    def _update(self, document: Document) -> None:
        """This method is called to update the metric with the new document."""

    @abstractmethod
    def _compute(self) -> T:
        """This method is called to get the metric values."""

    @property
    def current_split(self) -> Optional[str]:
        """The current split that is being processed."""
        return self._current_split
