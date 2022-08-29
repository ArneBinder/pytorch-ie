from typing import Generic, Iterator, Sequence, overload

from torch.utils.data.dataset import Dataset, IterableDataset

from pytorch_ie.core.taskmodule import DocumentType, InputEncoding, TargetEncoding, TaskEncoding


class TaskEncodingDataset(
    Dataset[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]],
    Generic[DocumentType, InputEncoding, TargetEncoding],
):
    def __init__(
        self, encodings: Sequence[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]]
    ):
        self._encodings = encodings

    def __getitem__(self, index) -> TaskEncoding[DocumentType, InputEncoding, TargetEncoding]:
        return self._encodings[index]

    def __len__(self):
        return len(self._encodings)


class IterableTaskEncodingDataset(
    IterableDataset[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]],
    Generic[DocumentType, InputEncoding, TargetEncoding],
):
    def __iter__(self) -> Iterator[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]]:
        yield from self._encodings

    def __init__(
        self, encodings: Iterator[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]]
    ):
        self._encodings = encodings


@overload
def as_dataset(
    encodings: Iterator[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]]
) -> IterableTaskEncodingDataset[DocumentType, InputEncoding, TargetEncoding]:
    ...


@overload
def as_dataset(
    encodings: Sequence[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]]
) -> TaskEncodingDataset[DocumentType, InputEncoding, TargetEncoding]:
    ...


def as_dataset(encodings):
    if isinstance(encodings, Iterator):
        return IterableTaskEncodingDataset(encodings=encodings)
    elif isinstance(encodings, Sequence):
        return TaskEncodingDataset(encodings=encodings)
    else:
        raise TypeError(f"encodings has unknown type")
