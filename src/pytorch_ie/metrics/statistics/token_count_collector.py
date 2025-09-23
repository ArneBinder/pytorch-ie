import logging
from typing import Any, Dict, Optional, Type, Union

from pie_core import Document, DocumentStatistic
from pie_documents.documents import TextBasedDocument
from transformers import AutoTokenizer, PreTrainedTokenizer

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
