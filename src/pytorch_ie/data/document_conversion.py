import functools
import logging
from collections import defaultdict
from copy import copy, deepcopy
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple, Type, TypeVar

from transformers import PreTrainedTokenizer

from pytorch_ie.annotations import Span
from pytorch_ie.core import Annotation
from pytorch_ie.documents import TextBasedDocument, TokenBasedDocument

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=TokenBasedDocument)


def text_based_document_to_token_based(
    doc: TextBasedDocument,
    tokens: List[str],
    result_document_type: Type[T],
    token_offset_mapping: Optional[List[Tuple[int, int]]] = None,
    char_to_token: Optional[Callable[[int], Optional[int]]] = None,
    strict_span_conversion: bool = True,
    verbose: bool = True,
) -> T:
    result = result_document_type(tokens=tuple(tokens), id=doc.id, metadata=deepcopy(doc.metadata))

    # save text, token_offset_mapping and char_to_token (if available) in metadata
    result.metadata["text"] = doc.text
    if token_offset_mapping is not None:
        result.metadata["token_offset_mapping"] = token_offset_mapping
    if char_to_token is not None:
        result.metadata["char_to_token"] = char_to_token

    # construct the char_to_token function, if not provided, from the token_offset_mapping
    if char_to_token is None:
        if token_offset_mapping is None:
            raise ValueError(
                "either token_offset_mapping or char_to_token must be provided to convert a text "
                "based document to token based, but both are None"
            )
        char_to_token_dict: Dict[int, int] = {}
        for token_idx, (start, end) in enumerate(token_offset_mapping):
            for char_idx in range(start, end):
                char_to_token_dict[char_idx] = token_idx

        def char_to_token(char_idx: int) -> Optional[int]:
            return char_to_token_dict.get(char_idx)

    text_span_layers = [
        annotation_field.name
        for annotation_field in doc.annotation_fields()
        if "text" in annotation_field.metadata["targets"]
    ]

    override_annotations: Dict[str, Dict[int, Annotation]] = {}
    removed_annotations: Dict[str, Set[int]] = defaultdict(set)
    for text_span_layer_name in text_span_layers:
        override_annotations[text_span_layer_name] = {}
        char_span: Span
        for char_span in doc[text_span_layer_name]:
            start_token_idx = char_to_token(char_span.start)
            end_token_idx_inclusive = char_to_token(char_span.end - 1)
            if start_token_idx is None or end_token_idx_inclusive is None:
                if strict_span_conversion:
                    raise ValueError(
                        f'cannot find token span for character span: "{char_span}", text="{doc.text}", '
                        f"token_offset_mapping={token_offset_mapping}"
                    )
                else:
                    if verbose:
                        logger.warning(
                            f'cannot find token span for character span "{char_span}", skip it (disable this '
                            f"warning with verbose=False)"
                        )
                    removed_annotations[text_span_layer_name].add(char_span._id)
            else:
                token_span = char_span.copy(start=start_token_idx, end=end_token_idx_inclusive + 1)
                override_annotations[text_span_layer_name][char_span._id] = token_span
        valid_spans = set(override_annotations[text_span_layer_name].values())
        result[text_span_layer_name].extend(sorted(valid_spans, key=lambda span: span.start))

    result.add_all_annotations_from_other(
        doc,
        override_annotations=override_annotations,
        removed_annotations=removed_annotations,
        strict=strict_span_conversion,
        verbose=verbose,
    )

    return result


def tokenize_document(
    doc: TextBasedDocument,
    tokenizer: PreTrainedTokenizer,
    result_document_type: Type[T],
    partition_layer: Optional[str] = None,
    strict_span_conversion: bool = True,
    verbose: bool = True,
    **tokenize_kwargs,
) -> List[T]:
    result = []
    partitions: Iterable[Span]
    if partition_layer is None:
        partitions = [Span(start=0, end=len(doc.text))]
    else:
        partitions = doc[partition_layer]
    for partition in partitions:
        text = doc.text[partition.start : partition.end]
        current_tokenize_kwargs = copy(tokenize_kwargs)
        if "text" in tokenize_kwargs:
            current_tokenize_kwargs["text_pair"] = text
            sequence_index = 1
        else:
            current_tokenize_kwargs["text"] = text
            sequence_index = 0
        tokenized_text = tokenizer(**current_tokenize_kwargs)
        for batch_encoding in tokenized_text.encodings:
            token_offset_mapping = batch_encoding.offsets
            char_to_token: Optional[Callable[[int], Optional[int]]]
            char_to_token = functools.partial(
                batch_encoding.char_to_token, sequence_index=sequence_index
            )
            token_offset_mapping = [
                offsets if s_id == sequence_index else (0, 0)
                for s_id, offsets in zip(batch_encoding.sequence_ids, token_offset_mapping)
            ]
            if partition.start > 0:
                token_offset_mapping = [
                    (start + partition.start, end + partition.start)
                    for start, end in token_offset_mapping
                ]
                char_to_token = None
            tokenized_document = text_based_document_to_token_based(
                doc,
                tokens=batch_encoding.tokens,
                result_document_type=result_document_type,
                token_offset_mapping=token_offset_mapping,
                char_to_token=char_to_token,
                strict_span_conversion=strict_span_conversion,
                verbose=verbose,
            )
            tokenized_document.metadata["tokenizer_encoding"] = batch_encoding
            result.append(tokenized_document)
    return result
