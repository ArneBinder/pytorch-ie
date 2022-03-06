import copy
import logging
from abc import ABC, abstractmethod
from typing import Dict, Generic, Iterator, List, Optional, Sequence, Tuple, TypeVar, Union

from pytorch_ie.core.hf_hub_mixin import PyTorchIETaskmoduleModelHubMixin
from pytorch_ie.data import Metadata
from pytorch_ie.data.document import Annotation, Document

"""
workflow:
    Document
        -> (InputEncoding, TargetEncoding) -> TaskEncoding -> TaskBatchEncoding
            -> ModelBatchEncoding -> ModelBatchOutput
        -> TaskOutput
    -> Document
"""

InputEncoding = TypeVar("InputEncoding")
TargetEncoding = TypeVar("TargetEncoding")
# TaskEncoding: defined below
TaskBatchEncoding = TypeVar("TaskBatchEncoding")
# ModelBatchEncoding: defined in models
ModelBatchOutput = TypeVar("ModelBatchOutput")
TaskOutput = TypeVar("TaskOutput")


logger = logging.getLogger(__name__)


class TaskEncoding(Generic[InputEncoding, TargetEncoding]):
    def __init__(
        self,
        input: InputEncoding,
        document: Document,
        target: Optional[TargetEncoding] = None,
        metadata: Optional[Metadata] = None,
    ) -> None:
        self.input = input
        self.document = document
        self._target = target
        self.metadata = metadata or {}

    @property
    def has_target(self) -> bool:
        return self._target is not None

    @property
    def target(self) -> TargetEncoding:
        # Note: mypy does not understand if we call self.has_target
        assert self._target is not None, "input encoding has no target"
        return self._target


class TaskModule(
    ABC,
    PyTorchIETaskmoduleModelHubMixin,
    Generic[InputEncoding, TargetEncoding, TaskBatchEncoding, ModelBatchOutput, TaskOutput],
):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

        assert len(input_encoding) == len(metadata) and len(input_encoding) == len(
            documents
        ), "'input_encoding', 'metadata', and 'documents' must be of same length."
        if target is None:
            return [
                TaskEncoding[InputEncoding, TargetEncoding](
                    input=enc_inp, metadata=md, document=doc
                )
                for enc_inp, md, doc in zip(input_encoding, metadata, documents)
            ]
        else:
            assert len(input_encoding) == len(
                target
            ), "'input_encoding' and 'target' must be of same length."
            return [
                TaskEncoding[InputEncoding, TargetEncoding](
                    input=enc_inp, document=doc, target=tgt, metadata=md
                )
                for enc_inp, md, tgt, doc in zip(input_encoding, metadata, target, documents)
            ]

    @abstractmethod
    def encode_input(
        self,
        documents: List[Document],
        is_training: bool = False,
    ) -> Tuple[List[InputEncoding], List[Metadata], Optional[List[Document]]]:
        raise NotImplementedError()

    @abstractmethod
    def encode_target(
        self,
        documents: List[Document],
        input_encodings: List[InputEncoding],
        metadata: List[Metadata],
    ) -> List[TargetEncoding]:
        raise NotImplementedError()

    @abstractmethod
    def unbatch_output(self, output: ModelBatchOutput) -> Sequence[TaskOutput]:
        """
        This method has to convert the batch output of the model (i.e. a dict of lists) to the list of individual
        outputs (i.e. a list of dicts). This is in preparation to generate a list of all model outputs that has the
        same length as all model inputs.
        """
        raise NotImplementedError()

    def decode(
        self,
        encodings: List[TaskEncoding[InputEncoding, TargetEncoding]],
        decoded_outputs: List[TaskOutput],
        input_documents: List[Document],
        inplace: bool = True,
    ) -> List[Document]:
        """
        This method takes the model inputs and (unbatched) model outputs and creates a list of documents that hold the
        new annotations created from model predictions.
        """
        if not inplace:
            copied_documents = {doc: copy.deepcopy(doc) for doc in input_documents}
            encodings = [
                TaskEncoding[InputEncoding, TargetEncoding](
                    input=encoding.input,
                    document=copied_documents[encoding.document],
                    target=encoding.target if encoding.has_target else None,
                    metadata=encoding.metadata,
                )
                for encoding in encodings
            ]
            documents = [copied_documents[doc] for doc in input_documents]
        else:
            documents = input_documents

        self.combine_outputs(encodings, decoded_outputs)
        return documents

    def combine_outputs(
        self,
        encodings: List[TaskEncoding[InputEncoding, TargetEncoding]],
        outputs: List[TaskOutput],
    ):
        for encoding, output in zip(encodings, outputs):
            self.combine_output(encoding=encoding, output=output)

    def combine_output(
        self,
        encoding: TaskEncoding[InputEncoding, TargetEncoding],
        output: TaskOutput,
    ):
        for annotation_name, annotation in self.create_annotations_from_output(
            encoding=encoding, output=output
        ):
            encoding.document.add_prediction(name=annotation_name, prediction=annotation)

    @abstractmethod
    def create_annotations_from_output(
        self,
        encoding: TaskEncoding[InputEncoding, TargetEncoding],
        output: TaskOutput,
    ) -> Iterator[Tuple[str, Annotation]]:
        raise NotImplementedError()

    @abstractmethod
    def collate(
        self, encodings: List[TaskEncoding[InputEncoding, TargetEncoding]]
    ) -> TaskBatchEncoding:
        raise NotImplementedError()
