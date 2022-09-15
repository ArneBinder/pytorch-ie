import collections.abc
import copy
import logging
from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import torch.utils.data.dataset as torch_dataset

from pytorch_ie.core.document import Annotation, Document
from pytorch_ie.core.hf_hub_mixin import PyTorchIETaskmoduleModelHubMixin
from pytorch_ie.core.registrable import Registrable
from pytorch_ie.data import Dataset, IterableDataset

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


Metadata = Dict[str, Any]


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


TaskEncodingType = TypeVar("TaskEncodingType", bound=TaskEncoding)


class TaskEncodingDataset(torch_dataset.Dataset[TaskEncodingType]):
    def __init__(self, encodings: Sequence[TaskEncodingType]):
        self._encodings = encodings

    @overload
    def __getitem__(self, index: int) -> TaskEncodingType:
        ...

    @overload
    def __getitem__(self, s: slice) -> Sequence[TaskEncodingType]:
        ...

    def __getitem__(
        self, index: Union[int, slice]
    ) -> Union[TaskEncodingType, Sequence[TaskEncodingType]]:
        return self._encodings[index]

    def __len__(self):
        return len(self._encodings)


class IterableTaskEncodingDataset(torch_dataset.IterableDataset[TaskEncodingType]):
    def __iter__(self) -> Iterator[TaskEncodingType]:
        yield from self._encodings

    def __init__(self, encodings: Iterator[TaskEncodingType]):
        self._encodings = encodings


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
    Registrable,
    Generic[
        DocumentType,
        InputEncoding,
        TargetEncoding,
        TaskBatchEncoding,
        ModelBatchOutput,
        TaskOutput,
    ],
):
    def __init__(self, encode_document_batch_size: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.encode_document_batch_size = encode_document_batch_size

    def _config(self) -> Dict[str, Any]:
        config = dict(self.hparams)
        this_class = self.__class__
        registered_name = TaskModule.registered_name_for_class(this_class)
        config["taskmodule_type"] = (
            registered_name if registered_name is not None else this_class.__name__
        )
        return config

    def prepare(self, documents: Sequence[DocumentType]) -> None:
        return None

    def batch_encode(
        self, documents: Union[Sequence[DocumentType], Dataset], encode_target: bool
    ) -> Tuple[
        Sequence[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]], Sequence[DocumentType]
    ]:
        ## TODO: revisit the assumption that encode_target=True always implies that
        ## is_training=True
        task_encodings, documents_in_order = self.encode_inputs(
            documents, is_training=encode_target
        )

        if encode_target:
            task_encodings = self.encode_targets(task_encodings)
        return task_encodings, documents_in_order

    def _encoding_iterator(
        self,
        documents: Iterable[DocumentType],
        encode_target: bool,
        batch_size: Optional[int] = None,
    ) -> Iterator[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]]:
        document_batch = []
        for i, doc in enumerate(documents):
            document_batch.append(doc)

            if batch_size is not None and len(document_batch) >= batch_size:
                yield from self.batch_encode(
                    documents=document_batch[:batch_size], encode_target=encode_target
                )[0]
                document_batch = document_batch[batch_size:]

        if len(document_batch) > 0:
            yield from self.batch_encode(documents=document_batch, encode_target=encode_target)[0]

    def encode(
        self,
        documents: Union[DocumentType, Sequence[DocumentType], Dataset, IterableDataset],
        encode_target: bool = False,
        document_batch_size: Optional[int] = None,
        as_task_encoding_sequence: Optional[bool] = None,
        as_iterator: Optional[bool] = None,
        as_dataset: bool = False,
    ) -> Union[
        Sequence[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]],
        TaskEncodingSequence[
            TaskEncoding[DocumentType, InputEncoding, TargetEncoding], DocumentType
        ],
        Iterator[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]],
        TaskEncodingDataset[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]],
        IterableTaskEncodingDataset[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]],
    ]:
        # backwards compatibility
        if as_task_encoding_sequence is None:
            as_task_encoding_sequence = not encode_target

        if not isinstance(documents, (Sequence, Dataset, IterableDataset)):
            documents = [documents]

        if as_iterator is None:
            as_iterator = isinstance(documents, (IterableDataset, Iterator))

        if document_batch_size is None:
            document_batch_size = self.encode_document_batch_size

        if as_iterator:
            if as_task_encoding_sequence:
                raise ValueError(f"can not return a TaskEncodingSequence as Iterator")
            encodings_iterator = self._encoding_iterator(
                documents=documents, encode_target=encode_target, batch_size=document_batch_size
            )
            if as_dataset:
                return IterableTaskEncodingDataset(encodings=encodings_iterator)
            else:
                return encodings_iterator
        else:
            encodings: List[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]] = []
            documents_in_order: List[DocumentType] = []
            docs_as_list = list(documents)
            bs = document_batch_size or len(docs_as_list)
            for i in range(0, len(docs_as_list), bs):
                cur_task_encodings, cur_documents_in_order = self.batch_encode(
                    documents=docs_as_list[i : i + bs], encode_target=encode_target
                )
                encodings.extend(cur_task_encodings)
                documents_in_order.extend(cur_documents_in_order)

            if as_task_encoding_sequence:
                if as_dataset:
                    raise ValueError(f"can not return a TaskEncodingSequence as a dataset")
                return TaskEncodingSequence(
                    task_encodings=encodings,
                    documents_in_order=documents_in_order,
                )
            else:
                # during training, we return only the sequence of task_encodings, because
                # we don't need the ordering of input documents and also don't re-assign
                # task encodings to input documents
                if as_dataset:
                    return TaskEncodingDataset(encodings=encodings)
                else:
                    return encodings

    def encode_inputs(
        self,
        documents: Union[Sequence[DocumentType], Dataset, IterableDataset],
        is_training: bool = False,
    ) -> Tuple[
        Sequence[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]],
        Sequence[DocumentType],
    ]:
        documents_in_order: List[DocumentType] = []
        task_encodings: List[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]] = []
        for document in documents:
            # a document might be generated on the fly (e.g. with a Dataset), so we add it here
            documents_in_order.append(document)

            possible_task_encodings = self.encode_input(document, is_training)

            # encode_input returns None or an empty list
            if possible_task_encodings is None or not possible_task_encodings:
                continue

            elif isinstance(possible_task_encodings, TaskEncoding):
                task_encodings.append(possible_task_encodings)

            else:
                task_encodings.extend(possible_task_encodings)

        return task_encodings, documents_in_order

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
    ) -> List[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]]:
        """
        Given a list of task encodings, get and assign the respective target encodings
        and return all task encodings that got a target.

        In that means, this will filter out all encodings without a target. This can be useful
        when different sets of encodings are required for training and inference. It mitigates the need
        to implement special logic that depends on target information in encode_input(). Encodings that
        are not suitable for training, i.e. where no target information is available, can be filtered out
        easily by letting encode_target() return None.
        """
        res = []
        for task_encoding in task_encodings:
            target_encoding = self.encode_target(task_encoding)
            if target_encoding is not None:
                task_encoding.targets = target_encoding
                res.append(task_encoding)
        return res

    @abstractmethod
    def encode_target(
        self,
        task_encoding: TaskEncoding[DocumentType, InputEncoding, TargetEncoding],
    ) -> Optional[TargetEncoding]:
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
        task_encodings: Union[
            Sequence[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]],
            TaskEncodingSequence[
                TaskEncoding[DocumentType, InputEncoding, TargetEncoding], DocumentType
            ],
        ],
        task_outputs: Sequence[TaskOutput],
        inplace: bool = True,
    ) -> Sequence[DocumentType]:
        """
        This method takes the model inputs and (unbatched) model outputs and creates a list of documents that hold the
        new annotations created from model predictions.
        """
        documents: Dict[int, DocumentType] = {}

        # TaskEncodingSequence provides us with the correct ordering
        if isinstance(task_encodings, TaskEncodingSequence):
            for document in task_encodings.documents_in_order:
                document_id = id(document)
                documents[document_id] = document if inplace else copy.deepcopy(document)
        # Otherwise we assume that documents are ordered according to the sequence of
        # unique documents defined by the sequence of task encodings
        else:
            for task_encoding in task_encodings:
                document = task_encoding.document
                document_id = id(document)
                if document_id not in documents:
                    documents[document_id] = document if inplace else copy.deepcopy(document)

        if not inplace:
            task_encodings = [
                TaskEncoding[DocumentType, InputEncoding, TargetEncoding](
                    document=documents[id(task_encoding.document)],
                    inputs=task_encoding.inputs,
                    targets=task_encoding.targets if task_encoding.has_targets else None,
                    metadata=task_encoding.metadata,
                )
                for task_encoding in task_encodings
            ]

        self.combine_outputs(task_encodings, task_outputs)

        unique_documents = list(documents.values())
        return unique_documents

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
