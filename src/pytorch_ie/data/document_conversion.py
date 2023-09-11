import functools
import logging
from collections import defaultdict
from copy import copy, deepcopy
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple, Type, TypeVar, Union

from transformers import PreTrainedTokenizer

from pytorch_ie.annotations import Span
from pytorch_ie.core import Annotation
from pytorch_ie.documents import TextBasedDocument, TokenBasedDocument
from pytorch_ie.utils.hydra import resolve_target

logger = logging.getLogger(__name__)

ToD = TypeVar("ToD", bound=TokenBasedDocument)
TeD = TypeVar("TeD", bound=TextBasedDocument)


def text_based_document_to_token_based(
    doc: TextBasedDocument,
    result_document_type: Union[Type[ToD], str],
    tokens: Optional[List[str]] = None,
    token_offset_mapping: Optional[List[Tuple[int, int]]] = None,
    char_to_token: Optional[Callable[[int], Optional[int]]] = None,
    strict_span_conversion: bool = True,
    verbose: bool = True,
) -> ToD:
    document_type: Type[ToD]
    if isinstance(result_document_type, str):
        document_type = resolve_target(result_document_type)  # type: ignore
    else:
        document_type = result_document_type
    if not (isinstance(document_type, type) and issubclass(document_type, TokenBasedDocument)):
        raise TypeError(
            f"result_document_type must be a subclass of TokenBasedDocument or a string that resolves to that, "
            f"but got {result_document_type}"
        )
    if tokens is None:
        tokens = doc.metadata.get("tokens")
    if tokens is None:
        raise ValueError(
            "tokens must be provided to convert a text based document to token based, but got None"
        )
    result = document_type(tokens=tuple(tokens), id=doc.id, metadata=deepcopy(doc.metadata))

    # save text, token_offset_mapping and char_to_token (if available) in metadata
    result.metadata["text"] = doc.text
    if token_offset_mapping is not None:
        if (
            "token_offset_mapping" in doc.metadata
            and doc.metadata["token_offset_mapping"] != token_offset_mapping
        ):
            logger.warning(
                "token_offset_mapping in metadata is different from the new token_offset_mapping, "
                "overwrite the metadata"
            )
        result.metadata["token_offset_mapping"] = token_offset_mapping
    else:
        token_offset_mapping = doc.metadata.get("token_offset_mapping")
    if char_to_token is not None:
        if "char_to_token" in doc.metadata and doc.metadata["char_to_token"] != char_to_token:
            logger.warning(
                "char_to_token in metadata is different from the new char_to_token, overwrite the metadata"
            )
        result.metadata["char_to_token"] = char_to_token
    else:
        char_to_token = doc.metadata.get("char_to_token")

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

    text_targeting_layers = [
        annotation_field.name
        for annotation_field in doc.annotation_fields()
        if "text" in annotation_field.metadata["targets"]
    ]

    override_annotations: Dict[str, Dict[int, Annotation]] = {}
    removed_annotations: Dict[str, Set[int]] = defaultdict(set)
    for text_targeting_layer_name in text_targeting_layers:
        override_annotations[text_targeting_layer_name] = {}
        char_span: Span
        for char_span in doc[text_targeting_layer_name]:
            if not isinstance(char_span, Span):
                raise ValueError(
                    f"can not convert layers that target the text but contain non-span annotations, "
                    f"but found {type(char_span)} in layer {text_targeting_layer_name}"
                )
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
                    removed_annotations[text_targeting_layer_name].add(char_span._id)
            else:
                token_span = char_span.copy(start=start_token_idx, end=end_token_idx_inclusive + 1)
                override_annotations[text_targeting_layer_name][char_span._id] = token_span
        valid_spans = set(override_annotations[text_targeting_layer_name].values())
        result[text_targeting_layer_name].extend(sorted(valid_spans, key=lambda span: span.start))

    result.add_all_annotations_from_other(
        doc,
        override_annotations=override_annotations,
        removed_annotations=removed_annotations,
        strict=strict_span_conversion,
        verbose=verbose,
    )

    return result


def token_based_document_to_text_based(
    doc: TokenBasedDocument,
    result_document_type: Union[Type[TeD], str],
    text: Optional[str] = None,
    token_offset_mapping: Optional[List[Tuple[int, int]]] = None,
    join_tokens_with: Optional[str] = None,
    strict_span_conversion: bool = True,
    verbose: bool = True,
) -> TeD:
    document_type: Type[TeD]
    if isinstance(result_document_type, str):
        document_type = resolve_target(result_document_type)  # type: ignore
    else:
        document_type = result_document_type
    if not (isinstance(document_type, type) and issubclass(document_type, TextBasedDocument)):
        raise TypeError(
            f"result_document_type must be a subclass of TextBasedDocument or a string that resolves to that, "
            f"but got {result_document_type}"
        )
    # if a token_separator is provided, we construct the text from the tokens
    if join_tokens_with is not None:
        start = 0
        token_offset_mapping = []
        tokens = doc.tokens
        for token in tokens:
            end = start + len(token)
            token_offset_mapping.append((start, end))
            # we add the separator after each token
            start = end + len(join_tokens_with)
        text = join_tokens_with.join(tokens)
    else:
        text = doc.metadata.get("text") if text is None else text
        if text is None:
            raise ValueError(
                "if join_tokens_with is None, text must be provided, but got None as well"
            )
        token_offset_mapping = (
            doc.metadata.get("token_offset_mapping")
            if token_offset_mapping is None
            else token_offset_mapping
        )
        if token_offset_mapping is None:
            raise ValueError(
                "if join_tokens_with is None, token_offsets must be provided, but got None as well"
            )

    result = document_type(text=text, id=doc.id, metadata=deepcopy(doc.metadata))
    if "tokens" in doc.metadata and doc.metadata["tokens"] != list(doc.tokens):
        logger.warning("tokens in metadata are different from new tokens, overwrite the metadata")
    result.metadata["tokens"] = list(doc.tokens)
    if (
        "token_offset_mapping" in doc.metadata
        and doc.metadata["token_offset_mapping"] != token_offset_mapping
    ):
        logger.warning(
            "token_offset_mapping in metadata is different from the new token_offset_mapping, "
            "overwrite the metadata"
        )
    result.metadata["token_offset_mapping"] = token_offset_mapping

    token_targeting_layers = [
        annotation_field.name
        for annotation_field in doc.annotation_fields()
        if "tokens" in annotation_field.metadata["targets"]
    ]

    override_annotations: Dict[str, Dict[int, Annotation]] = {}
    removed_annotations: Dict[str, Set[int]] = defaultdict(set)
    for token_targeting_layer_name in token_targeting_layers:
        override_annotations[token_targeting_layer_name] = {}
        for token_span in doc[token_targeting_layer_name]:
            if not isinstance(token_span, Span):
                raise ValueError(
                    f"can not convert layers that target the tokens but contain non-span annotations, "
                    f"but found {type(token_span)} in layer {token_targeting_layer_name}"
                )
            start_char_idx = token_offset_mapping[token_span.start][0]
            end_char_idx = token_offset_mapping[token_span.end - 1][1]

            char_span = token_span.copy(start=start_char_idx, end=end_char_idx)
            override_annotations[token_targeting_layer_name][token_span._id] = char_span
        valid_spans = set(override_annotations[token_targeting_layer_name].values())
        result[token_targeting_layer_name].extend(sorted(valid_spans, key=lambda span: span.start))

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
    result_document_type: Type[ToD],
    partition_layer: Optional[str] = None,
    strict_span_conversion: bool = True,
    verbose: bool = True,
    **tokenize_kwargs,
) -> List[ToD]:
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
