import functools
import json
import logging
from collections import defaultdict
from copy import copy
from typing import Callable, Dict, Hashable, Iterable, List, Optional, Type, TypeVar

from pie_core import Annotation, Document
from pie_core.document import BaseAnnotationList
from pie_documents.annotations import Span
from pie_documents.document.processing import text_based_document_to_token_based
from pie_documents.documents import TextBasedDocument, TokenBasedDocument, WithMetadata
from transformers import BatchEncoding, PreTrainedTokenizer

logger = logging.getLogger(__name__)


def deduplicate_annotation_dicts(
    annotation_dicts: List[Dict[str, Hashable]]
) -> List[Dict[str, Hashable]]:
    """Remove duplicate annotation dictionaries from a list of annotation dictionaries.

    Args:
        annotation_dicts: The list of annotation dictionaries to remove duplicates from.

    Returns:
        The list of annotation dictionaries with duplicates removed.
    """
    unique_annotation_dicts = []
    seen_annotation_dicts = set()
    for annotation_dict in annotation_dicts:
        annotation_dict_tuple = tuple(sorted(annotation_dict.items()))
        if annotation_dict_tuple not in seen_annotation_dicts:
            unique_annotation_dicts.append(annotation_dict)
            seen_annotation_dicts.add(annotation_dict_tuple)
    return unique_annotation_dicts


D = TypeVar("D", bound=Document)


def save_annotation_sources_to_metadata(
    document: D,
    annotation_id2source: Dict[int, List[str]],
    metadata_key: str,
    use_predictions: bool,
) -> None:
    """Save the source names for the annotations or predictions in the metadata of the document.

    Args:
        document: The document to save the source names in the metadata for.
        metadata_key: The key in the metadata where the source names should be stored.
        annotation_id2source: A mapping from annotation IDs to the source names. Should contain
            the ids of all annotations or predictions (depending on use_predictions) in the document.
        use_predictions: Whether to store the source names for the predictions or the annotations.
    """

    if not hasattr(document, "metadata"):
        raise ValueError("Document does not have metadata, can not store source names.")
    if metadata_key in document.metadata:
        raise ValueError(f"Metadata key '{metadata_key}' already exists in the document.")
    document.metadata[metadata_key] = defaultdict(dict)
    for annotation_field in document.annotation_fields():
        layer_name = annotation_field.name
        document.metadata[metadata_key][layer_name] = []
        layer: BaseAnnotationList
        if use_predictions:
            layer = document[layer_name].predictions
        else:
            layer = document[layer_name]
        for ann in layer:
            document.metadata[metadata_key][layer_name].append(annotation_id2source[ann._id])
    document.metadata[metadata_key] = dict(document.metadata[metadata_key])


def merge_annotations_from_documents(
    documents: Dict[str, D],
    metadata_key_source_annotations: Optional[str] = None,
    metadata_key_source_predictions: Optional[str] = None,
) -> D:
    """Merge annotations from multiple documents into a single document. Optionally, store the source
    names for all annotations / predictions in the metadata at key metadata_key_source_annotations
    / metadata_key_source_predictions, respectively.

    Note that this will remove any annotation duplicates.

    Args:
        documents: A dictionary mapping document source (e.g. dataset names) to documents.
        metadata_key_source_annotations: If not None, the key in the metadata where the source names
            for the (gold) annotations are stored.
        metadata_key_source_predictions: If not None, the key in the metadata where the source names
            for the predictions are stored.

    Returns:
        The document with merged annotations.
    """
    if len(documents) == 0:
        raise ValueError("No documents provided.")
    source_names = sorted(documents)
    first_source_name = source_names[0]
    merged_document: D = documents[first_source_name].copy(with_annotations=False)

    added_annotation_id2source_names: Dict[int, List[str]] = defaultdict(list)
    for source_name in source_names:
        document = documents[source_name]
        if type(document) is not type(merged_document):
            raise ValueError(
                f"Document types do not match: {type(document)} and {type(merged_document)}"
            )
        if isinstance(document, WithMetadata) and document.id is not None:
            if document.id != merged_document.id:
                raise ValueError(
                    f"Document IDs do not match: {document.id} and {merged_document.id}"
                )

        # Note: this does not check for duplicates!
        added_annotations = merged_document.add_all_annotations_from_other(
            other=document, strict=True
        )

        for layer_name, orig_id2new_annotation in added_annotations.items():
            for orig_id, new_annotation in orig_id2new_annotation.items():
                added_annotation_id2source_names[new_annotation._id].append(source_name)

    # this will remove duplicates. If duplicates have different scores, the one with the highest score will be kept
    merged_document = merged_document.deduplicate_annotations()

    # save source names in metadata (at key metadata_key_source_annotations / metadata_key_source_predictions
    #   for each layer in the order of the annotations / predictions)
    if metadata_key_source_annotations is not None:
        save_annotation_sources_to_metadata(
            document=merged_document,
            annotation_id2source=added_annotation_id2source_names,
            metadata_key=metadata_key_source_annotations,
            use_predictions=False,
        )
    if metadata_key_source_predictions is not None:
        save_annotation_sources_to_metadata(
            document=merged_document,
            annotation_id2source=added_annotation_id2source_names,
            metadata_key=metadata_key_source_predictions,
            use_predictions=True,
        )
    return merged_document


ToD = TypeVar("ToD", bound=TokenBasedDocument)


def tokenize_document(
    doc: TextBasedDocument,
    tokenizer: PreTrainedTokenizer,
    result_document_type: Type[ToD],
    partition_layer: Optional[str] = None,
    strip_spans: bool = False,
    strict_span_conversion: bool = True,
    added_annotations: Optional[List[Dict[str, Dict[Annotation, Annotation]]]] = None,
    verbose: bool = True,
    **tokenize_kwargs,
) -> List[ToD]:
    """Tokenize a document with a given tokenizer and return a list of token based documents. The
    document is tokenized in partitions if a partition layer is provided. The annotations that
    target the text are converted to target the tokens and also all dependent annotations are
    converted.

    Args:
        doc (TextBasedDocument): The document to tokenize.
        tokenizer (PreTrainedTokenizer): The tokenizer.
        result_document_type (Type[ToD]): The exact type of the token based documents.
        partition_layer (Optional[str], optional): The layer to use for partitioning the document. If None, the whole
            document is tokenized. Defaults to None.
        strip_spans (bool, optional): If True, strip the whitespace from the character spans before converting them to
            token spans. Defaults to False.
        strict_span_conversion (bool, optional): If True, raise an error if not all annotations can be converted to
            token based documents. Defaults to True.
        added_annotations (Optional[List[Dict[str, Dict[Annotation, Annotation]]]], optional): Pass an empty list to
            collect the added annotations. Defaults to None.
        verbose (bool, optional): If True, log warnings if annotations can not be converted. Defaults to True.

    Returns:
        List[ToD]: The token based documents of type result_document_type with the converted annotations.
    """

    added_annotation_lists: Dict[str, List[Annotation]] = defaultdict(list)
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
            current_added_annotations: Dict[str, Dict[Annotation, Annotation]] = defaultdict(dict)
            tokenized_document = text_based_document_to_token_based(
                doc,
                tokens=batch_encoding.tokens,
                result_document_type=result_document_type,
                token_offset_mapping=token_offset_mapping,
                char_to_token=char_to_token,
                strict_span_conversion=False,
                strip_spans=strip_spans,
                verbose=False,
                added_annotations=current_added_annotations,
            )
            tokenized_document.metadata["tokenizer_encoding"] = batch_encoding
            result.append(tokenized_document)
            for k, v in current_added_annotations.items():
                added_annotation_lists[k].extend(v)
            if added_annotations is not None:
                added_annotations.append(current_added_annotations)

    missed_annotations = defaultdict(set)
    if strict_span_conversion or verbose:
        # We check the annotations with respect to the layers of the result_document_type.
        # Note that the original document may have more layers, but since result documents
        # are of type result_document_type, we only check the layers of this type.
        for annotation_field in result_document_type.annotation_fields():
            # do not check the partition layer because the partitions are not required later on
            # and entries get quite probably removed when windowing is applied, so this just pollutes the logs
            if annotation_field.name != partition_layer:
                current_missed_annotations = set(doc[annotation_field.name]) - set(
                    added_annotation_lists[annotation_field.name]
                )
                if len(current_missed_annotations) > 0:
                    missed_annotations[annotation_field.name] = current_missed_annotations

    if len(missed_annotations) > 0:
        missed_annotations_simplified = {k: str(v) for k, v in missed_annotations.items()}
        if strict_span_conversion:
            raise ValueError(
                f"could not convert all annotations from document with id={doc.id} to token based documents, "
                f"but strict_span_conversion is True, so raise an error, "
                f"missed annotations:\n{json.dumps(missed_annotations_simplified, sort_keys=True, indent=2)}"
            )
        else:
            if verbose:
                logger.warning(
                    f"could not convert all annotations from document with id={doc.id} to token based documents, "
                    f"missed annotations (disable this message with verbose=False):\n"
                    f"{json.dumps(missed_annotations_simplified, sort_keys=True, indent=2)}"
                )

    return result


S = TypeVar("S", bound=Span)


class SpanNotAlignedWithTokenException(Exception):
    def __init__(self, span):
        self.span = span


def get_aligned_token_span(encoding: BatchEncoding, char_span: S) -> S:
    # find the start
    token_start = None
    token_end_before = None
    char_start = None
    for idx in range(char_span.start, char_span.end):
        token_start = encoding.char_to_token(idx)
        if token_start is not None:
            char_start = idx
            break

    if char_start is None:
        raise SpanNotAlignedWithTokenException(span=char_span)
    for idx in range(char_span.end - 1, char_start - 1, -1):
        token_end_before = encoding.char_to_token(idx)
        if token_end_before is not None:
            break

    if token_start is None or token_end_before is None:
        raise SpanNotAlignedWithTokenException(span=char_span)

    return char_span.copy(start=token_start, end=token_end_before + 1)
