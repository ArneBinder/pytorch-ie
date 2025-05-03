from collections.abc import Iterator, Sequence
from typing import TypeVar, Union, overload

import torch.utils.data.dataset as torch_dataset
from pie_core import TaskEncoding

TaskEncodingType = TypeVar("TaskEncodingType", bound=TaskEncoding)


class TaskEncodingDataset(torch_dataset.Dataset[TaskEncodingType]):
    def __init__(self, encodings: Sequence[TaskEncodingType]):
        self._encodings = encodings

    @overload
    def __getitem__(self, index: int) -> TaskEncodingType: ...

    @overload
    def __getitem__(self, s: slice) -> Sequence[TaskEncodingType]: ...

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
