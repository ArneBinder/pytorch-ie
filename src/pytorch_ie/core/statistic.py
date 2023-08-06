from abc import abstractmethod
from collections import defaultdict
from typing import Any, Dict, Generator, List, Tuple, Union

from pytorch_ie.core.document import Document
from pytorch_ie.core.metric import DocumentMetric


def _flatten_dict_gen(d, parent_key: Tuple[str, ...] = ()) -> Generator:
    for k, v in d.items():
        new_key = parent_key + (k,)
        if isinstance(v, dict):
            yield from dict(_flatten_dict_gen(v, new_key)).items()
        else:
            yield new_key, v


def flatten_dict(d: Dict[str, Any]) -> Dict[Tuple[str, ...], Any]:
    return dict(_flatten_dict_gen(d))


def unflatten_dict(d: Dict[Tuple[str, ...], Any]) -> Union[Dict[str, Any], Any]:
    """Unflattens a dictionary with nested keys.

    Example:
        >>> d = {("a", "b", "c"): 1, ("a", "b", "d"): 2, ("a", "e"): 3}
        >>> unflatten_dict(d)
        {'a': {'b': {'c': 1, 'd': 2}, 'e': 3}}
    """
    result: Dict[str, Any] = {}
    for k, v in d.items():
        if len(k) == 0:
            if len(result) > 1:
                raise ValueError("Cannot unflatten dictionary with multiple root keys.")
            return v
        current = result
        for key in k[:-1]:
            current = current.setdefault(key, {})
        current[k[-1]] = v
    return result


class DocumentStatistic(DocumentMetric):
    """A special type of metric that collects statistics from a document.

    Usage:

    ```python
    from transformers import AutoTokenizer
    from pytorch_ie import DatasetDict

    class DocumentTokenCounter(DocumentStatistic):
        def __init__(self, tokenizer_name_or_path: str, field: str, **kwargs):
            super().__init__()
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
            self.kwargs = kwargs
            self.field = field

        def _collect(self, doc: Document) -> int:
            text = getattr(doc, self.field)
            encodings = self.tokenizer(text, **self.kwargs)
            tokens = encodings.tokens()
            return len(tokens)

    dataset = DatasetDict.load_dataset("pie/conll2003")
    statistic = DocumentTokenCounter(tokenizer_name_or_path="bert-base-cased", field="text")
    values = statistic(dataset)
    ```
    """

    def reset(self) -> None:
        self._values: List[Any] = []

    @abstractmethod
    def _collect(self, doc: Document) -> Any:
        """Collect any values from a document."""

    def _update(self, document: Document) -> None:
        values = self._collect(document)
        self._values.append(values)

    def _compute(self) -> Any:
        """We just integrate the values by creating lists for each leaf of the (nested)
        dictionary."""
        stats = defaultdict(list)
        for metric_result in self._values:
            if isinstance(metric_result, dict):
                measure_result_flat = flatten_dict(metric_result)
                for k, v in measure_result_flat.items():
                    if isinstance(v, list):
                        stats[k].extend(v)
                    else:
                        stats[k].append(v)
            else:
                if isinstance(metric_result, list):
                    stats[()].extend(metric_result)
                else:
                    stats[()].append(metric_result)
        return unflatten_dict(dict(stats))
