import logging
from typing import Optional, Type

from pytorch_ie.core.document import Document
from pytorch_ie.data.dataset_dict import DatasetDict

logger = logging.getLogger(__name__)


class RequiresDocumentTypeMixin:

    DOCUMENT_TYPE: Optional[Type[Document]] = None

    @property
    def document_type(self) -> Optional[Type[Document]]:
        return self.DOCUMENT_TYPE

    def convert_dataset(self, dataset: DatasetDict) -> DatasetDict:
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
