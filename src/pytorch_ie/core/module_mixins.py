import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Type

from pytorch_ie.core.document import Document

logger = logging.getLogger(__name__)


class WithDocumentTypeMixin:

    DOCUMENT_TYPE: Optional[Type[Document]] = None

    @property
    def document_type(self) -> Optional[Type[Document]]:
        return self.DOCUMENT_TYPE

    def convert_dataset(self, dataset: "pie_datasets.DatasetDict") -> "pie_datasets.DatasetDict":  # type: ignore
        name = type(self).__name__
        # auto-convert the dataset if a document type is specified
        if self.document_type is not None:
            if issubclass(dataset.document_type, self.document_type):
                logger.info(
                    f"the dataset is already of the document type that is specified by {name}: "
                    f"{self.document_type}"
                )
            else:
                logger.info(
                    f"convert the dataset to the document type that is specified by {name}: "
                    f"{self.document_type}"
                )
                dataset = dataset.to_document_type(self.document_type)
        else:
            logger.warning(
                f"{name} does not specify a document type. The dataset can not be automatically converted "
                f"to a document type."
            )

        return dataset


class PreparableMixin:
    # list of attribute names that need to be set by _prepare()
    PREPARED_ATTRIBUTES: List[str] = []

    @property
    def is_prepared(self):
        """
        Returns True, iff all attributes listed in PREPARED_ATTRIBUTES are set.
        Note: Attributes set to None are not considered to be prepared!
        """
        return all(
            getattr(self, attribute, None) is not None for attribute in self.PREPARED_ATTRIBUTES
        )

    @property
    def prepared_attributes(self):
        if not self.is_prepared:
            raise Exception("The module is not prepared.")
        return {param: getattr(self, param) for param in self.PREPARED_ATTRIBUTES}

    def _prepare(self, *args, **kwargs):
        """
        This method needs to set all attributes listed in PREPARED_ATTRIBUTES.
        """
        pass

    def _post_prepare(self):
        """
        Any code to do further one-time setup, but that requires the prepared attributes.
        """
        pass

    def assert_is_prepared(self, msg: Optional[str] = None):
        if not self.is_prepared:
            attributes_not_prepared = [
                param for param in self.PREPARED_ATTRIBUTES if getattr(self, param, None) is None
            ]
            raise Exception(
                f"{msg or ''} Required attributes that are not set: {str(attributes_not_prepared)}"
            )

    def post_prepare(self):
        self.assert_is_prepared()
        self._post_prepare()

    def prepare(self, *args, **kwargs) -> None:
        if self.is_prepared:
            if len(self.PREPARED_ATTRIBUTES) > 0:
                msg = f"The {self.__class__.__name__} is already prepared, do not prepare again."
                for k, v in self.prepared_attributes.items():
                    msg += f"\n{k} = {str(v)}"
                logger.warning(msg)
        else:
            self._prepare(*args, **kwargs)
            self.assert_is_prepared(
                msg=f"_prepare() was called, but the {self.__class__.__name__} is not prepared."
            )
        self._post_prepare()
        return None


class EnterDatasetMixin(ABC):
    """Mixin for processors that enter a dataset context."""

    @abstractmethod
    def enter_dataset(self, dataset, name: Optional[str] = None) -> None:
        """Enter dataset context."""


class ExitDatasetMixin(ABC):
    """Mixin for processors that exit a dataset context."""

    @abstractmethod
    def exit_dataset(self, dataset, name: Optional[str] = None) -> None:
        """Exit dataset context."""


class EnterDatasetDictMixin(ABC):
    """Mixin for processors that enter a dataset dict context."""

    @abstractmethod
    def enter_dataset_dict(self, dataset_dict) -> None:
        """Enter dataset dict context."""


class ExitDatasetDictMixin(ABC):
    """Mixin for processors that exit a dataset dict context."""

    @abstractmethod
    def exit_dataset_dict(self, dataset_dict) -> None:
        """Exit dataset dict context."""
