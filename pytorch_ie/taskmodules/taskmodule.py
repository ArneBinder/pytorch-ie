import copy
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, Generic, TypeVar

from pytorch_ie.core.hf_hub_mixin import PyTorchIETaskmoduleModelHubMixin
from pytorch_ie.data.document import Annotation, AnnotationCollection, Document

InputEncoding = TypeVar('InputEncoding', bound=Dict[str, Any])
TargetEncoding = TypeVar('TargetEncoding', bound=Dict[str, Any])
ModelOutput = TypeVar('ModelOutput', bound=Dict[str, Any])
Metadata = Dict[str, Any]
BatchedModelOutput = Dict[str, Any]

logger = logging.getLogger(__name__)


class TaskEncoding(Generic[InputEncoding, TargetEncoding]):
    def __init__(
        self,
        input: InputEncoding,
        document: Document,
        target: Optional[TargetEncoding] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.input = input
        self.document = document
        self.target = target
        self.metadata = metadata or {}


class TaskModule(ABC, PyTorchIETaskmoduleModelHubMixin, Generic[InputEncoding, TargetEncoding, ModelOutput]):
    def __init__(self, **kwargs):
        self.init_inputs = ()
        self.init_kwargs = copy.deepcopy(kwargs)

    def prepare(self, documents: List[Document]) -> None:
        return

    def encode(
        self, documents: Union[Document, List[Document]], encode_target: bool = False
    ) -> List[TaskEncoding[InputEncoding, TargetEncoding]]:
        if isinstance(documents, Document):
            documents = [documents]

        input_encoding, metadata, new_documents = self.encode_input(documents)

        if new_documents is not None:
            documents = new_documents

        target = None
        if encode_target:
            target = self.encode_target(documents, input_encoding, metadata)

        if target is None:
            assert len(input_encoding) == len(metadata) and len(input_encoding) == len(
                documents
            ), "'input_encoding', 'metadata', and 'documents' must be of same length."
            return [
                TaskEncoding(input=enc_inp, metadata=md, document=doc)
                for enc_inp, md, doc in zip(input_encoding, metadata, documents)
            ]

        assert (
            len(input_encoding) == len(metadata)
            and len(input_encoding) == len(target)
            and len(input_encoding) == len(documents)
        ), "'input_encoding', 'metadata', 'target', and 'documents' must be of same length."

        return [
            TaskEncoding(input=enc_inp, document=doc, target=tgt, metadata=md)
            for enc_inp, md, tgt, doc in zip(input_encoding, metadata, target, documents)
        ]

    @abstractmethod
    def encode_input(
        self, documents: List[Document]
    ) -> Tuple[List[InputEncoding], Optional[List[Metadata]], Optional[List[Document]]]:
        raise NotImplementedError()

    @abstractmethod
    def encode_target(
        self,
        documents: List[Document],
        input_encodings: List[InputEncoding],
        metadata: Optional[List[Metadata]],
    ) -> List[TargetEncoding]:
        raise NotImplementedError()

    @abstractmethod
    def unbatch_output(self, output: BatchedModelOutput) -> List[ModelOutput]:
        """
        This method has to convert the batch output of the model (i.e. a dict of lists) to the list of individual
        outputs (i.e. a list of dicts). This is in preparation to generate a list of all model outputs that has the
        same length as all model inputs.
        """
        raise NotImplementedError()

    def decode(
        self,
        encodings: List[TaskEncoding],
        decoded_outputs: List[ModelOutput],
        inplace: bool = True,
    ) -> List[Document]:
        """
        This method takes the model inputs and (unbatched) model outputs and creates a list of documents that hold the
        new annotations created from model predictions.
        """
        if not inplace:
            copied_documents: Dict[Document, Document] = {}
            copied_encodings: List[TaskEncoding] = []
            for encoding in encodings:
                if encoding.document not in copied_documents:
                    copied_documents[encoding.document] = copy.deepcopy(encoding.document)

                copied_encodings.append(
                    TaskEncoding(
                        input=encoding.input,
                        document=copied_documents[encoding.document],
                        target=encoding.target,
                        metadata=encoding.metadata,
                    )
                )
            all_documents = list(copied_documents.values())
            encodings = copied_encodings
        else:
            all_documents = list(set((encoding.document for encoding in encodings)))

        self.combine_outputs(encodings, decoded_outputs)
        return all_documents

    def combine_outputs(
        self,
        encodings: List[TaskEncoding],
        outputs: List[ModelOutput],
    ):
        for encoding, output in zip(encodings, outputs):
            self.combine_output(encoding=encoding, output=output)

    def combine_output(
        self,
        encoding: TaskEncoding,
        output: ModelOutput,
    ):
        for annotation_name, annotation in self.create_annotations_from_output(
            encoding=encoding, output=output
        ):
            encoding.document.add_prediction(name=annotation_name, prediction=annotation)

    def create_annotations_from_output(
        self,
        encoding: TaskEncoding,
        output: ModelOutput,
    ) -> Iterator[Tuple[str, Annotation]]:
        raise NotImplementedError()

    @abstractmethod
    def collate(self, encodings: List[TaskEncoding]) -> Dict[str, Any]:
        raise NotImplementedError()
