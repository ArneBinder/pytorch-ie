import collections.abc
import copy
import logging
from abc import ABC, abstractmethod
from typing import (
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

from pytorch_ie import Dataset, Document
from pytorch_ie.annotations import Annotation
from pytorch_ie.core.hf_hub_mixin import PyTorchIETaskmoduleModelHubMixin
from pytorch_ie.data import Metadata

"""
workflow:
    Document
        -> (InputEncoding, TargetEncoding) -> TaskEncoding -> TaskBatchEncoding
            -> ModelBatchEncoding -> ModelBatchOutput
        -> TaskOutput
    -> Document
"""

DocumentType = TypeVar("DocumentType", bound=Document)
InputEncoding = TypeVar("InputEncoding")
TargetEncoding = TypeVar("TargetEncoding")
# TaskEncoding: defined below
TaskBatchEncoding = TypeVar("TaskBatchEncoding")
# ModelBatchEncoding: defined in models
ModelBatchOutput = TypeVar("ModelBatchOutput")
TaskOutput = TypeVar("TaskOutput")


logger = logging.getLogger(__name__)


class InplaceNotSupportedException(Exception):
    pass


class TaskEncoding(Generic[DocumentType, InputEncoding, TargetEncoding]):
    def __init__(
        self,
        document: DocumentType,
        inputs: InputEncoding,
        targets: Optional[TargetEncoding] = None,
        metadata: Optional[Metadata] = None,
    ) -> None:
        self.document = document
        self.inputs = inputs
        self._targets = targets
        self.metadata = metadata or {}
        self._doc_idx: Optional[int] = None

    @property
    def has_targets(self) -> bool:
        return self._targets is not None

    @property
    def targets(self) -> TargetEncoding:
        # TODO: find a better solution
        assert self._targets is not None, "task encoding has no targets"
        return self._targets

    @targets.setter
    def targets(self, value) -> None:
        self._targets = value

    @property
    def doc_idx(self) -> int:
        if self._doc_idx is None:
            raise Exception(f"doc_idx is not set")
        return self._doc_idx

    @doc_idx.setter
    def doc_idx(self, value: int):
        self._doc_idx = value


TaskEncodingType = TypeVar("TaskEncodingType", bound=TaskEncoding)


class TaskEncodingSequence(
    collections.abc.Sequence[TaskEncodingType], Generic[TaskEncodingType, DocumentType]
):
    def __init__(
        self,
        task_encodings: Sequence[TaskEncodingType],
        documents_in_order: Sequence[DocumentType],
    ):
        self.task_encodings = task_encodings
        self.documents_in_order = documents_in_order

    @overload
    def __getitem__(self, index: int) -> TaskEncodingType:
        ...

    @overload
    def __getitem__(self, s: slice) -> Sequence[TaskEncodingType]:
        ...

    def __getitem__(
        self, index: Union[int, slice]
    ) -> Union[TaskEncodingType, Sequence[TaskEncodingType]]:
        return self.task_encodings[index]

    def __len__(self) -> int:
        return len(self.task_encodings)


class TaskModule(
    ABC,
    PyTorchIETaskmoduleModelHubMixin,
    Generic[
        DocumentType,
        InputEncoding,
        TargetEncoding,
        TaskBatchEncoding,
        ModelBatchOutput,
        TaskOutput,
    ],
):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def prepare(self, documents: Sequence[DocumentType]) -> None:
        return None

    def encode(
        self,
        documents: Union[DocumentType, Sequence[DocumentType], Dataset],
        encode_target: bool = False,
    ) -> Sequence[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]]:
        if not isinstance(documents, (Sequence, Dataset)):
            documents = [documents]

        task_encodings = self.encode_inputs(documents, is_training=encode_target)

        if encode_target:
            self.encode_targets(task_encodings)

        return task_encodings

    def encode_inputs(
        self,
        documents: Union[Sequence[DocumentType], Dataset],
        is_training: bool = False,
    ) -> Sequence[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]]:
        task_encodings: List[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]] = []

        for idx, document in enumerate(documents):

            possible_task_encodings = self.encode_input(document, is_training)
            current_encodings = []

            # encode_input returns None or an empty list
            if possible_task_encodings is None or not possible_task_encodings:
                continue
            elif isinstance(possible_task_encodings, TaskEncoding):
                current_encodings.append(possible_task_encodings)
            else:
                current_encodings.extend(possible_task_encodings)

            # remember the document index since the document id may change to allow correct decoding
            for encoding in current_encodings:
                encoding.doc_idx = idx

            task_encodings.extend(current_encodings)

        return task_encodings

    @abstractmethod
    def encode_input(
        self,
        document: DocumentType,
        is_training: bool = False,
    ) -> Optional[
        Union[
            TaskEncoding[DocumentType, InputEncoding, TargetEncoding],
            Sequence[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]],
        ]
    ]:
        pass

    def encode_targets(
        self,
        task_encodings: Sequence[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]],
    ) -> None:
        for task_encoding in task_encodings:
            target_encoding = self.encode_target(task_encoding)
            task_encoding.targets = target_encoding

    @abstractmethod
    def encode_target(
        self,
        task_encoding: TaskEncoding[DocumentType, InputEncoding, TargetEncoding],
    ) -> TargetEncoding:
        pass

    @abstractmethod
    def unbatch_output(self, model_output: ModelBatchOutput) -> Sequence[TaskOutput]:
        """
        This method has to convert the batch output of the model (i.e. a dict of lists) to the list of individual
        outputs (i.e. a list of dicts). This is in preparation to generate a list of all model outputs that has the
        same length as all model inputs.
        """
        pass

    def decode(
        self,
        task_encodings: Sequence[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]],
        task_outputs: Sequence[TaskOutput],
        documents_in_encode_order: Union[Sequence[Document], Dataset],
        inplace: bool = True,
    ) -> Sequence[DocumentType]:
        """
        This method takes the model inputs and (unbatched) model outputs and creates a list of documents that hold the
        new annotations created from model predictions.
        """
        encode_idx_to_document: Dict[int, DocumentType] = {
            idx: doc if inplace else copy.deepcopy(doc)
            for idx, doc in enumerate(documents_in_encode_order)
        }

        if inplace:
            for task_encoding in task_encodings:
                document = task_encoding.document
                encode_idx_to_document[task_encoding.doc_idx] = document
        else:
            task_encodings = [
                TaskEncoding[DocumentType, InputEncoding, TargetEncoding](
                    document=encode_idx_to_document[task_encoding.doc_idx],
                    inputs=task_encoding.inputs,
                    targets=task_encoding.targets if task_encoding.has_targets else None,
                    metadata=task_encoding.metadata,
                )
                for task_encoding in task_encodings
            ]

        self.combine_outputs(task_encodings, task_outputs)

        return [encode_idx_to_document[idx] for idx in sorted(encode_idx_to_document)]

    def combine_outputs(
        self,
        task_encodings: Sequence[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]],
        task_outputs: Sequence[TaskOutput],
    ):
        for task_encoding, task_output in zip(task_encodings, task_outputs):
            self.combine_output(task_encoding, task_output)

    def combine_output(
        self,
        task_encoding: TaskEncoding[DocumentType, InputEncoding, TargetEncoding],
        task_output: TaskOutput,
    ):
        for annotation_name, annotation in self.create_annotations_from_output(
            task_encoding, task_output
        ):
            task_encoding.document[annotation_name].predictions.append(annotation)

    @abstractmethod
    def create_annotations_from_output(
        self,
        task_encoding: TaskEncoding[DocumentType, InputEncoding, TargetEncoding],
        task_output: TaskOutput,
    ) -> Iterator[Tuple[str, Annotation]]:
        pass

    @abstractmethod
    def collate(
        self, task_encodings: Sequence[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]]
    ) -> TaskBatchEncoding:
        pass
