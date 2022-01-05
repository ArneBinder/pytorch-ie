import copy
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator

from pytorch_ie.core.hf_hub_mixin import PyTorchIETaskmoduleModelHubMixin
from pytorch_ie.data.document import Document, Annotation, AnnotationCollection

InputEncoding = Dict[str, Any]
Metadata = Dict[str, Any]
TargetEncoding = Dict[str, Any]
ModelOutput = Dict[str, Any]
DecodedModelOutput = Dict[str, Any]

logger = logging.getLogger(__name__)


class TaskEncoding:
    def __init__(
        self,
        input: Dict[str, Any],
        document: Document,
        target: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.input = input
        self.document = document
        self.target = target
        self.metadata = metadata or {}


class TaskModule(ABC, PyTorchIETaskmoduleModelHubMixin):
    def __init__(self, **kwargs):
        self.init_inputs = ()
        self.init_kwargs = copy.deepcopy(kwargs)

    def prepare(self, documents: List[Document]) -> None:
        return

    def encode(
        self, documents: Union[Document, List[Document]], encode_target: bool = False
    ) -> List[TaskEncoding]:
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

    def decode(self, output: ModelOutput) -> List[DecodedModelOutput]:
        return self.decode_output(output)

    @abstractmethod
    def decode_output(self, output: ModelOutput) -> List[DecodedModelOutput]:
        raise NotImplementedError()

    def combine(
        self,
        encodings: List[TaskEncoding],
        decoded_outputs: List[DecodedModelOutput],
        inplace: bool = True,
    ) -> List[Document]:
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
        for annotation_name, annotation in self.decoded_output_to_annotations(encoding=encoding, output=output):
            encoding.document.add_prediction(name=annotation_name, prediction=annotation)

    def decoded_output_to_annotations(
        self,
        encoding: TaskEncoding,
        output: DecodedModelOutput,
    ) -> Iterator[Tuple[str, Annotation]]:
        raise NotImplementedError()

    @abstractmethod
    def collate(self, encodings: List[TaskEncoding]) -> Dict[str, Any]:
        raise NotImplementedError()
