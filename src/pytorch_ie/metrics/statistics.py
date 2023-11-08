import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Type, Union

from transformers import AutoTokenizer, PreTrainedTokenizer

from pytorch_ie.core import Document, DocumentStatistic
from pytorch_ie.documents import TextBasedDocument

logger = logging.getLogger(__name__)


class TokenCountCollector(DocumentStatistic):
    """Collects the token count of a field when tokenizing its content with a Huggingface
    tokenizer.

    The content of the field should be a string.
    """

    def __init__(
        self,
        tokenizer: Union[str, PreTrainedTokenizer],
        text_field: str = "text",
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        document_type: Optional[Type[Document]] = None,
        **kwargs,
    ):
        if document_type is None and text_field == "text":
            document_type = TextBasedDocument
        super().__init__(document_type=document_type, **kwargs)
        self.tokenizer = (
            AutoTokenizer.from_pretrained(tokenizer) if isinstance(tokenizer, str) else tokenizer
        )
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.text_field = text_field

    def _collect(self, doc: Document) -> int:
        text = getattr(doc, self.text_field)
        encodings = self.tokenizer(text, **self.tokenizer_kwargs)
        tokens = encodings.tokens()
        return len(tokens)


class FieldLengthCollector(DocumentStatistic):
    """Collects the length of a field, e.g. to collect the number the characters in the input text.

    The field should be a list of sized elements.
    """

    def __init__(self, field: str, **kwargs):
        super().__init__(**kwargs)
        self.field = field

    def _collect(self, doc: Document) -> int:
        field_obj = getattr(doc, self.field)
        return len(field_obj)


class SubFieldLengthCollector(DocumentStatistic):
    """Collects the length of a subfield in a field, e.g. to collect the number of arguments of
    N-ary relations."""

    def __init__(self, field: str, subfield: str, **kwargs):
        super().__init__(**kwargs)
        self.field = field
        self.subfield = subfield

    def _collect(self, doc: Document) -> List[int]:
        field_obj = getattr(doc, self.field)
        lengths = []
        for entry in field_obj:
            subfield_obj = getattr(entry, self.subfield)
            lengths.append(len(subfield_obj))
        return lengths


class DummyCollector(DocumentStatistic):
    """A dummy collector that always returns 1, e.g. to count the number of documents.

    Can be used to count the number of documents.
    """

    DEFAULT_AGGREGATION_FUNCTIONS = ["sum"]

    def _collect(self, doc: Document) -> int:
        return 1


class LabelCountCollector(DocumentStatistic):
    """Collects the number of field entries per label, e.g. to collect the number of entities per
    type.

    The field should be a list of elements with a label attribute.

    Important: To make correct use of the result data, missing values need to be filled with 0, e.g.:
        {("ORG",): [2, 3], ("LOC",): [2]} -> {("ORG",): [2, 3], ("LOC",): [2, 0]}
    """

    DEFAULT_AGGREGATION_FUNCTIONS = ["mean", "std", "min", "max", "len", "sum"]

    def __init__(
        self, field: str, labels: Union[List[str], str], label_attribute: str = "label", **kwargs
    ):
        super().__init__(**kwargs)
        self.field = field
        self.label_attribute = label_attribute
        if not (isinstance(labels, list) or labels == "INFERRED"):
            raise ValueError("labels must be a list of strings or 'INFERRED'")
        if labels == "INFERRED":
            logger.warning(
                f"Inferring labels with {self.__class__.__name__} from data produces wrong results "
                f"for certain aggregation functions (e.g. 'mean', 'std', 'min') because zero values "
                f"are not included in the calculation. We remove these aggregation functions from "
                f"this collector, but be aware that the results may be wrong for your own aggregation "
                f"functions that rely on zero values."
            )
            self.aggregation_functions: Dict[str, Callable[[List], Any]] = {
                name: func
                for name, func in self.aggregation_functions.items()
                if name not in ["mean", "std", "min"]
            }

        self.labels = labels

    def _collect(self, doc: Document) -> Dict[str, int]:
        field_obj = getattr(doc, self.field)
        counts: Dict[str, int]
        if self.labels == "INFERRED":
            counts = defaultdict(int)
        else:
            counts = {label: 0 for label in self.labels}
        for elem in field_obj:
            label = getattr(elem, self.label_attribute)
            counts[label] += 1
        return dict(counts)
