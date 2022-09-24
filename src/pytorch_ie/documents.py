import dataclasses
from typing import Any, Dict, Optional, Tuple

from pytorch_ie.annotations import OcrAnnotation, OcrLabeledSpan
from pytorch_ie.core import AnnotationList, Document, annotation_field


@dataclasses.dataclass
class TextDocument(Document):
    text: str
    id: Optional[str] = None
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class OcrDocument(Document):
    # 3d: channel x row x col
    image: Tuple[
        Tuple[Tuple[int, ...], ...], Tuple[Tuple[int, ...], ...], Tuple[Tuple[int, ...], ...]
    ]
    image_width: int
    image_height: int
    image_format: str
    words: AnnotationList[OcrAnnotation] = annotation_field(target="image")

    def __post_init__(self):
        # when creating from a dataset, this comes in as a list (json does not know tuples)
        if not isinstance(self.image, tuple):
            object.__setattr__(
                self,
                "image",
                tuple(tuple(tuple(row) for row in channel) for channel in self.image),
            )
        super().__post_init__()


@dataclasses.dataclass
class OcrDocumentWithEntities(OcrDocument):
    entities: AnnotationList[OcrLabeledSpan] = annotation_field(target="words")
