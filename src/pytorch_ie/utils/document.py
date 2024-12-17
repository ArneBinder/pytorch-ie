from collections import defaultdict
from typing import Dict, Hashable, List, Optional, TypeVar

from pytorch_ie.core.document import BaseAnnotationList, Document
from pytorch_ie.documents import WithMetadata


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
