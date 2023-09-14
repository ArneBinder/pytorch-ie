from collections import defaultdict
from typing import Any, Dict, List, Optional, Type, Union

from transformers import AutoTokenizer, PreTrainedTokenizer

from pytorch_ie.core import Document, DocumentStatistic
from pytorch_ie.documents import TextBasedDocument


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


class LabeledSpanLengthCollector(DocumentStatistic):
    """Collects the length of spans in a field per label, e.g. to collect the length of entities per type.

    The field should be a list of elements with a label, a start and end attribute.
    """

    DEFAULT_AGGREGATION_FUNCTIONS = ["mean", "std", "min", "max", "len"]

    def __init__(self, field: str, **kwargs):
        super().__init__(**kwargs)
        self.field = field

    def _collect(self, doc: Document) -> Dict[str, List[int]]:
        field_obj = getattr(doc, self.field)
        counts = defaultdict(list)
        for elem in field_obj:
            counts[elem.label].append(elem.end - elem.start)
        return dict(counts)


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

    DEFAULT_AGGREGATION_FUNCTIONS = ["mean", "std", "min", "max", "len"]

    def __init__(self, field: str, labels: List[str], **kwargs):
        super().__init__(**kwargs)
        self.field = field
        self.labels = labels

    def _collect(self, doc: Document) -> Dict[str, int]:
        field_obj = getattr(doc, self.field)
        counts: Dict[str, int] = {label: 0 for label in self.labels}
        for elem in field_obj:
            counts[elem.label] += 1
        return dict(counts)
