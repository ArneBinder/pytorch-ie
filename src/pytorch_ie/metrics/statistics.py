import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Type, Union

from transformers import AutoTokenizer, PreTrainedTokenizer

from pytorch_ie import tokenize_document
from pytorch_ie.annotations import Span
from pytorch_ie.core import Document, DocumentStatistic
from pytorch_ie.documents import TextBasedDocument, TokenBasedDocument
from pytorch_ie.utils.hydra import resolve_optional_document_type

logger = logging.getLogger(__name__)


class TokenCountCollector(DocumentStatistic):
    """Collects the token count of a field when tokenizing its content with a Huggingface tokenizer.

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
    """Collects the length of a subfield in a field, e.g. to collect the number of arguments of N-ary relations."""

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


class SpanLengthCollector(DocumentStatistic):
    """Collects the lengths of Span annotations. If labels are provided, the lengths collected per
    label.

    If a tokenizer is provided, the span length is calculated in means of tokens, otherwise in
    means of characters.
    """

    DEFAULT_AGGREGATION_FUNCTIONS = ["len", "mean", "std", "min", "max"]

    def __init__(
        self,
        layer: str,
        tokenize: bool = False,
        tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
        tokenized_document_type: Optional[Union[str, Type[TokenBasedDocument]]] = None,
        labels: Optional[Union[List[str], str]] = None,
        label_attribute: str = "label",
        tokenize_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.layer = layer
        if isinstance(labels, str) and labels != "INFERRED":
            raise ValueError("labels must be a list of strings or 'INFERRED'")
        if labels == "INFERRED":
            logger.warning(
                f"Inferring labels with {self.__class__.__name__} from data produces wrong results "
                f"for certain aggregation functions (e.g. 'mean', 'std', 'min') because zero values "
                f"are not included in the calculation. We remove these aggregation functions from "
                f"this collector, but be aware that the results may be wrong for your own aggregation "
                f"functions that rely on zero values."
            )
            self.aggregation_functions = {
                name: func
                for name, func in self.aggregation_functions.items()
                if name not in ["mean", "std", "min"]
            }
        self.labels = labels
        self.label_field = label_attribute
        self.tokenize = tokenize
        if self.tokenize:
            if tokenizer is None:
                raise ValueError(
                    "tokenizer must be provided to calculate the span length in means of tokens"
                )
            if isinstance(tokenizer, str):
                tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            self.tokenizer = tokenizer
            resolved_tokenized_document_type = resolve_optional_document_type(
                tokenized_document_type
            )
            if resolved_tokenized_document_type is None:
                raise ValueError(
                    "tokenized_document_type must be provided to calculate the span length in means of tokens"
                )
            if not (
                isinstance(resolved_tokenized_document_type, type)
                and issubclass(resolved_tokenized_document_type, TokenBasedDocument)
            ):
                raise TypeError(
                    f"tokenized_document_type must be a subclass of TokenBasedDocument, but it is: "
                    f"{resolved_tokenized_document_type}"
                )
            self.tokenized_document_type = resolved_tokenized_document_type
            self.tokenize_kwargs = tokenize_kwargs or {}

    def _collect(self, doc: Document) -> Union[List[int], Dict[str, List[int]]]:
        docs: Union[List[Document], List[TokenBasedDocument]]
        if self.tokenize:
            if not isinstance(doc, TextBasedDocument):
                raise ValueError(
                    "doc must be a TextBasedDocument to calculate the span length in means of tokens"
                )
            if not isinstance(doc, TextBasedDocument):
                raise ValueError(
                    "doc must be a TextBasedDocument to calculate the span length in means of tokens"
                )
            docs = tokenize_document(
                doc,
                tokenizer=self.tokenizer,
                result_document_type=self.tokenized_document_type,
                **self.tokenize_kwargs,
            )
        else:
            docs = [doc]

        values: Dict[str, List[int]]
        if isinstance(self.labels, str):
            values = defaultdict(list)
        else:
            values = {label: [] for label in self.labels or ["ALL"]}
        for doc in docs:
            layer_obj = getattr(doc, self.layer)
            for span in layer_obj:
                if not isinstance(span, Span):
                    raise TypeError(
                        f"span length calculation is not yet supported for {type(span)}"
                    )
                length = span.end - span.start
                if self.labels is None:
                    label = "ALL"
                else:
                    label = getattr(span, self.label_field)
                values[label].append(length)

        return values if self.labels is not None else values["ALL"]


class DummyCollector(DocumentStatistic):
    """A dummy collector that always returns 1, e.g. to count the number of documents.

    Can be used to count the number of documents.
    """

    DEFAULT_AGGREGATION_FUNCTIONS = ["sum"]

    def _collect(self, doc: Document) -> int:
        return 1


class LabelCountCollector(DocumentStatistic):
    """Collects the number of field entries per label, e.g. to collect the number of entities per type.

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
            self.aggregation_functions = {
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
