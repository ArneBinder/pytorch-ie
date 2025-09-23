import dataclasses
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, Generic, Iterable, List, Optional, Tuple, TypeVar

import pandas as pd
import torch
import torch.nn.functional as F
from pie_core import Annotation
from torch import Tensor

logger = logging.getLogger(__name__)


def _pad_tensor(tensor: Tensor, target_shape: List[int], pad_value: float) -> Tensor:
    """Pad a tensor to a target shape.

    Args:
        tensor: The tensor to pad.
        target_shape: The target shape.
        pad_value: The value to use for padding.

    Returns: The padded tensor.
    """

    shape = tensor.shape
    pad: List[int] = []
    for i, s in enumerate(shape):
        pad = [0, target_shape[i] - s] + pad
    result = F.pad(tensor, pad=pad, value=pad_value)

    return result


def maybe_pad_values(
    values: Any, pad_value: Optional[Any] = None, strategy: str = "longest"
) -> Any:
    """If an iterable of values is passed and a pad value is given, pad the values to the same
    length and create a tensor from them. Otherwise, return the values unchanged.

    Note that the padding is done on all dimensions.

    Args:
        values: The values to pad.
        pad_value: The value to use for padding.
        strategy: The padding strategy. Currently only "longest" is supported.

    Returns: The padded values.
    """

    if pad_value is None:
        return values
    if not isinstance(values, Iterable):
        raise TypeError(f"values must be iterable to pad them, but got {type(values)}")
    if strategy != "longest":
        raise ValueError(f"unknown padding strategy: {strategy}")
    tensor_list = [torch.tensor(value_list) for value_list in values]
    shape_lists = list(zip(*[t.shape for t in tensor_list]))
    max_shape = [max(dims) for dims in shape_lists]
    padded = [
        _pad_tensor(tensor=t, target_shape=max_shape, pad_value=pad_value)
        for i, t in enumerate(tensor_list)
    ]
    return torch.stack(padded)


def maybe_to_tensor(
    values: Iterable[Any], dtype: Optional[torch.dtype] = None, pad_value: Optional[Any] = None
) -> Any:
    """If an iterable of values is passed and a dtype is given, convert the values to a tensor of
    the given type.

    Args:
        values: The values to convert.
        dtype: A dtype to convert the values to.
        pad_value: A pad value to use if the values are padded.

    Returns: A tensor or the values unchanged.
    """

    if all(v is None for v in values):
        return None
    if dtype is None:
        return values
    maybe_padded = maybe_pad_values(values=values, pad_value=pad_value)
    if not isinstance(maybe_padded, torch.Tensor):
        maybe_padded = torch.Tensor(maybe_padded)
    tensor = maybe_padded.to(dtype=dtype)
    return tensor


class BatchableMixin:
    """A mixin class that provides a batch method to batch a list of instances of the class. All
    attributes, but also property methods, are batched. The batch method returns a dictionary with
    all attribute / property names as keys. The values are tensors created from the stacked values
    of the attributes / properties. The tensors are padded to the length of the longest instance in
    the batch and converted to the given dtype.

    Example:
        >>> import dataclasses
        >>> from typing import List, Dict
        >>> import torch
        >>>
        >>> @dataclasses.dataclass
        >>> class Foo(BatchableMixin):
        >>>     a: List[int]
        >>>
        >>>   @property
        >>>   def len_a(self):
        >>>       return len(self.a)
        >>>
        >>> x = Foo(a=[1, 2, 3])
        >>> y = Foo(a=[4, 5])
        >>>
        >>> Foo.batch(values=[x, y], dtypes={"a": torch.int64, "len_a": torch.int64}, pad_values={"a": 0})
        {'a': tensor([[1, 2, 3],[4, 5, 0]]), 'len_a': tensor([3, 2])}
    """

    @classmethod
    def get_property_names(cls) -> List[str]:
        return [name for name in cls.__dict__ if isinstance(getattr(cls, name), property)]

    @classmethod
    def get_dataclass_field_names(cls) -> List[str]:
        if dataclasses.is_dataclass(cls):
            return [f.name for f in dataclasses.fields(cls)]
        else:
            return []

    @classmethod
    def get_attribute_names(cls) -> List[str]:
        return cls.get_property_names() + cls.get_dataclass_field_names()

    @classmethod
    def batch(
        cls,
        values: List[Any],
        dtypes: Dict[str, torch.dtype],
        pad_values: Dict[str, Any],
    ) -> Dict[str, Any]:
        attribute_names = cls.get_attribute_names()
        return {
            k: maybe_to_tensor(
                values=[getattr(x, k) for x in values],
                dtype=dtypes.get(k, None),
                pad_value=pad_values.get(k, None),
            )
            for k in attribute_names
            # Only batch attributes that are not None for any of the values.
            if not all(getattr(x, k) is None for x in values)
        }


T = TypeVar("T")


def increase_counter(
    key: Tuple[Any, ...],
    statistics: Dict[Tuple[Any, ...], int],
    value: int = 1,
):
    key_s = tuple(str(k) for k in key)
    statistics[key_s] += value


class StatisticsMixin(ABC, Generic[T]):
    """A mixin class that provides methods to collect and format statistics.

    Args:
        collect_statistics: Control whether statistics should be collected.
            If `False`, the mixin will not show any statistics when calling
            `show_statistics`. Further effects depend on the implementation
            of the mixin.
        **kwargs: Additional keyword arguments to pass to the parent class.
    """

    def __init__(self, collect_statistics: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.collect_statistics = collect_statistics
        self.reset_statistics()

    @abstractmethod
    def reset_statistics(self):
        """Reset the statistics collected by this mixin (state)."""
        pass

    @abstractmethod
    def get_statistics(self) -> T:
        """Get the statistics collected by this mixin.

        This should *not* modify the state of the mixin, repeated calls should return the same
        result!
        """
        pass

    def format_statistics(self, statistics: T) -> str:
        """Format the statistics collected by this mixin as string for display (usually on
        console)."""
        raise NotImplementedError(
            f"format_statistics is not implemented for {self.__class__.__name__}. "
            "Please implement this method to show formatted statistics."
        )

    def show_statistics(self):
        if self.collect_statistics:
            logger.info(f"statistics:\n{self.format_statistics(self.get_statistics())}")


class RelationStatisticsMixin(StatisticsMixin[Dict[Tuple[str, str], int]]):
    """A mixin class that provides methods to collect and format statistics about relations.

    This mixin collects statistics about relations, such as the number of available, used, and
    skipped relations.
    """

    def get_none_label_for_statistics(self) -> str:
        if not hasattr(self, "_statistics_none_label"):
            if hasattr(self, "none_label"):
                # If the mixin has a `none_label` attribute, use it as the label for "no relation".
                self._statistics_none_label = self.none_label
            else:
                self._statistics_none_label = "no_relation"
                logger.warning(
                    f"{type(self).__name__} does not have a `none_label` attribute. "
                    "Using default value 'no_relation'. "
                    "`none_label` is used as the label for relations with score 0 in statistics and "
                    "all relations with label different from `none_label` will be summarized to 'all_relations'. "
                    "Set the `none_label` attribute before using statistics or "
                    "overwrite `get_none_label_for_statistics()` function to get rid of this message."
                )

        return self._statistics_none_label

    def reset_statistics(self):
        self._collected_relations: Dict[str, List[Annotation]] = defaultdict(list)

    def collect_relation(self, kind: str, relation: Annotation):
        if self.collect_statistics:
            self._collected_relations[kind].append(relation)

    def collect_all_relations(self, kind: str, relations: Iterable[Annotation]):
        if self.collect_statistics:
            self._collected_relations[kind].extend(relations)

    def get_statistics(self) -> Dict[Tuple[str, str], int]:
        if self.collect_statistics:
            # create statistics from the collected relations
            statistics: Dict[Tuple[str, str], int] = defaultdict(int)
            all_relations = set(self._collected_relations["available"])
            used_relations = set(self._collected_relations["used"])
            skipped_other = all_relations - used_relations
            for key, rels in self._collected_relations.items():
                rels_set = set(rels)
                if key.startswith("skipped_"):
                    skipped_other -= rels_set
                elif key.startswith("used_"):
                    pass
                elif key in ["available", "used"]:
                    pass
                else:
                    raise ValueError(f"unknown key: {key}")
                for rel in rels_set:
                    # Set `none_label` as label when the score is zero. We encode negative relations
                    # in such a way in the case of multi-label or binary (similarity for coref).
                    label = rel.label if rel.score > 0 else self.get_none_label_for_statistics()
                    increase_counter(key=(key, label), statistics=statistics)
            for rel in skipped_other:
                increase_counter(key=("skipped_other", rel.label), statistics=statistics)

            return dict(statistics)
        else:
            return {}

    def format_statistics(self, statistics: Dict[Tuple[str, str], int]) -> str:
        if len(statistics) > 0:
            to_show_series = pd.Series(statistics)
            # unstack index to have relation labels as column names
            to_show = to_show_series.unstack()
        else:
            # If there were no statistics, create an empty dummy dataframe.
            to_show = pd.DataFrame(pd.Series(dict()))
        # fill missing values with 0 and convert back to int (unstacking may introduce NaNs which are float type)
        to_show = to_show.fillna(0).astype(int)
        if to_show.columns.size > 1:
            to_show["all_relations"] = to_show.loc[
                :, to_show.columns != self.get_none_label_for_statistics()
            ].sum(axis=1)

        #  transpose
        #  to have the labels (which may be a lot) as index for improved readability and
        #  to allow to keep counts as int columns (dtypes are per-column, not per-row)
        to_show = to_show.T
        if "used" in to_show.columns and "available" in to_show.columns:
            to_show["used %"] = (100 * to_show["used"] / to_show["available"]).round()

        return to_show.to_markdown()
