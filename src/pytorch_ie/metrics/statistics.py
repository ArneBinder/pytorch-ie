from collections import defaultdict
from typing import Dict, List

from transformers import AutoTokenizer

from pytorch_ie.core import Document, DocumentStatistic


class TokenCountCollector(DocumentStatistic):
    """Collects the token count of a field when tokenizing its content with a Huggingface tokenizer.

    The field should be a string.
    """

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


class FieldLengthCollector(DocumentStatistic):
    """Collects the length of a field, e.g. to collect the number the characters in the input text.

    The field should be a list of sized elements.
    """

    def __init__(self, field: str):
        super().__init__()
        self.field = field

    def _collect(self, doc: Document) -> int:
        field_obj = getattr(doc, self.field)
        return len(field_obj)


class SubFieldLengthCollector(DocumentStatistic):
    """Collects the length of a subfield in a field, e.g. to collect the number of arguments of N-ary relations."""

    def __init__(self, field: str, subfield: str):
        super().__init__()
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

    def __init__(self, field: str):
        super().__init__()
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

    def _collect(self, doc: Document) -> int:
        return 1


class LabelCountCollector(DocumentStatistic):
    """Collects the number of field entries per label, e.g. to collect the number of entities per type.

    The field should be a list of elements with a label attribute.

    Important: To make correct use of the result data, missing values need to be filled with 0, e.g.:
        {("ORG",): [2, 3], ("LOC",): [2]} -> {("ORG",): [2, 3], ("LOC",): [2, 0]}
    """

    def __init__(self, field: str):
        super().__init__()
        self.field = field

    def _collect(self, doc: Document) -> Dict[str, int]:
        field_obj = getattr(doc, self.field)
        counts: Dict[str, int] = defaultdict(lambda: 1)
        for elem in field_obj:
            counts[elem.label] += 1
        return dict(counts)
