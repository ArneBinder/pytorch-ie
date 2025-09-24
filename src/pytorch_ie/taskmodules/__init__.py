from .cross_text_binary_coref import CrossTextBinaryCorefTaskModule
from .labeled_span_extraction_by_token_classification import (
    LabeledSpanExtractionByTokenClassificationTaskModule,
)
from .pointer_network_for_end2end_re import PointerNetworkTaskModuleForEnd2EndRE
from .re_text_classification_with_indices import RETextClassificationWithIndicesTaskModule
from .simple_transformer_text_classification import SimpleTransformerTextClassificationTaskModule
from .text_to_text import TextToTextTaskModule
from .transformer_re_text_classification import TransformerRETextClassificationTaskModule
from .transformer_seq2seq import TransformerSeq2SeqTaskModule
from .transformer_span_classification import TransformerSpanClassificationTaskModule
from .transformer_text_classification import TransformerTextClassificationTaskModule
from .transformer_token_classification import TransformerTokenClassificationTaskModule

__all__ = [
    "SimpleTransformerTextClassificationTaskModule",
    "TransformerRETextClassificationTaskModule",
    "TransformerSeq2SeqTaskModule",
    "TransformerSpanClassificationTaskModule",
    "TransformerTextClassificationTaskModule",
    "TransformerTokenClassificationTaskModule",
    "RETextClassificationWithIndicesTaskModule",
    "TextToTextTaskModule",
    "LabeledSpanExtractionByTokenClassificationTaskModule",
    "PointerNetworkTaskModuleForEnd2EndRE",
    "CrossTextBinaryCorefTaskModule",
]
