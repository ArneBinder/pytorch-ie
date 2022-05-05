from typing import List

import pyarrow as pa
from datasets.formatting.formatting import Formatter

from pytorch_ie.core.document import Document


class DocumentFormatter(Formatter[Document, list, List[Document]]):
    def __init__(self, document_type, features=None, decoded=True, **kwargs):
        super().__init__(features=None, decoded=None)
        self.document_type = document_type

    def format_row(self, pa_table: pa.Table) -> Document:
        row = self.python_arrow_extractor().extract_row(pa_table)
        return self.document_type.fromdict(row)

    def format_column(self, pa_table: pa.Table) -> list:
        return []

    def format_batch(self, pa_table: pa.Table) -> List[Document]:
        batch = self.simple_arrow_extractor().extract_batch(pa_table).to_pylist()
        return [self.document_type.fromdict(b) for b in batch]
