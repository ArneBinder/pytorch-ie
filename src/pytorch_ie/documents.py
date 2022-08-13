import dataclasses
from typing import Any, Dict, Optional

from pytorch_ie.core import Document


@dataclasses.dataclass
class TextDocument(Document):
    text: str
    id: Optional[str] = None
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)
