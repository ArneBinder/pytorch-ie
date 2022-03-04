import hashlib
import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

from datasets import DatasetDict, load_dataset

from pytorch_ie import Document
from pytorch_ie.data.datasets import HF_DATASETS_ROOT
from pytorch_ie.data.document import Annotation, BinaryRelation, LabeledMultiSpan, LabeledSpan

DEFAULT_HEAD_ARGUMENT_NAME: str = "Arg1"
DEFAULT_TAIL_ARGUMENT_NAME: str = "Arg2"
DEFAULT_SPAN_ANNOTATION_NAME: str = "entities"
DEFAULT_RELATION_ANNOTATION_NAME: str = "relations"

GLUE_TEXT = "\n"
GLUE_BRAT = " "

logger = logging.getLogger(__name__)


def dl_to_ld(dl):
    return [dict(zip(dl, t)) for t in zip(*dl.values())]


# not used
def ld_to_dl(ld):
    return {k: [dic[k] for dic in ld] for k in ld[0]}


def hash_string(s: str, max_length: Optional[int] = None):
    hash_object = hashlib.sha256(s.encode())
    hex_dig = hash_object.hexdigest()
    if max_length is not None:
        hex_dig = hex_dig[-max_length:]
    return hex_dig


def convert_brat_to_document(
    brat_doc: Dict[str, Any],
    head_argument_name: str = DEFAULT_HEAD_ARGUMENT_NAME,
    tail_argument_name: str = DEFAULT_TAIL_ARGUMENT_NAME,
    span_annotation_name: str = DEFAULT_SPAN_ANNOTATION_NAME,
    relation_annotation_name: str = DEFAULT_RELATION_ANNOTATION_NAME,
    convert_multi_spans: bool = True,
) -> Document:

    doc = Document(text=brat_doc["context"], doc_id=brat_doc["file_name"])

    # add spans
    doc.annotations.spans.create_layer(name=span_annotation_name)
    span_id_mapping = {}
    for brat_span in dl_to_ld(brat_doc["spans"]):
        locations = dl_to_ld(brat_span["locations"])
        label = brat_span["type"]
        # strip annotation type identifier from id
        metadata = {"text": brat_span["text"], "id": brat_span["id"][1:]}
        span: LabeledMultiSpan
        if convert_multi_spans:
            if len(locations) > 1:
                added_fragments = [
                    brat_doc["context"][locations[i]["end"] : locations[i + 1]["start"]]
                    for i in range(len(locations) - 1)
                ]
                added_fragments_filtered = [frag for frag in added_fragments if frag != GLUE_TEXT]
                if len(added_fragments_filtered) > 0:
                    logger.warning(
                        f"convert span with several slices to LabeledSpan! added text fragments: "
                        f"{added_fragments_filtered}"
                    )
            span = LabeledSpan(
                start=locations[0]["start"],
                end=locations[-1]["end"],
                label=label,
                metadata=metadata,
            )
        else:
            span = LabeledMultiSpan(
                slices=[(location["start"], location["end"]) for location in locations],
                label=label,
                metadata=metadata,
            )
        assert (
            brat_span["id"] not in span_id_mapping
        ), f'brat span id "{brat_span["id"]}" already exists'
        span_id_mapping[brat_span["id"]] = span
        doc.add_annotation(name=span_annotation_name, annotation=span)

    # add relations
    doc.annotations.binary_relations.create_layer(name=relation_annotation_name)
    for brat_relation in dl_to_ld(brat_doc["relations"]):
        # strip annotation type identifier from id
        metadata = {"id": brat_relation["id"][1:]}
        brat_args = {arg["type"]: arg["target"] for arg in dl_to_ld(brat_relation["arguments"])}
        head = span_id_mapping[brat_args[head_argument_name]]
        tail = span_id_mapping[brat_args[tail_argument_name]]
        assert isinstance(head, LabeledSpan) and isinstance(tail, LabeledSpan), (
            f"BinaryRelation does only except head and tail of type `{LabeledSpan.__name__}`, "
            f"but they have types `{type(head)}` and `{type(tail)}`."
        )
        relation = BinaryRelation(
            label=brat_relation["type"], head=head, tail=tail, metadata=metadata
        )
        doc.add_annotation(name=relation_annotation_name, annotation=relation)

    # add events -> not yet implement
    # add equivalence_relations -> not yet implement
    # add attributions -> not yet implement
    # add normalizations -> not yet implement
    # add notes -> not yet implement

    return doc


def _convert_brats(brat_docs: Iterable[Dict[str, Any]], **kwargs) -> List[Document]:
    return [convert_brat_to_document(brat_doc=brat_doc, **kwargs) for brat_doc in brat_docs]


def load_brat(
    conversion_kwargs: Optional[Dict[str, Any]] = None,
    train_test_split: Dict[str, Any] = None,
    **kwargs,
) -> Dict[str, List[Document]]:
    # This will create a DatasetDict with a single split "train"
    path = str(HF_DATASETS_ROOT / "brat.py")
    data: DatasetDict = load_dataset(path=path, **kwargs)  # type: ignore
    assert isinstance(data, DatasetDict), (
        f"`load_dataset` for `path={path}` should return a DatasetDict (at least containing a single 'train' split), "
        f"but the result is of type: {type(data)}"
    )
    logger.info(f"loaded BRAT documents: " + str({k: len(v) for k, v in data.items()}))

    if train_test_split is not None:
        assert len(data) == 1, (
            f"Can only create a train test split from a single input split, but the loaded data contains "
            f"multiple splits: {', '.join(data)}. You may consider `subdirectory_mapping` to select one of them."
        )
        split_all = list(data.values())[0]
        data = split_all.train_test_split(**train_test_split)
    conversion_kwargs = conversion_kwargs or {}
    return {
        split: _convert_brats(brat_docs=brat_dataset, **conversion_kwargs)
        for split, brat_dataset in data.items()
    }


def split_span_annotation(text: str, slice: Tuple[int, int], glue: str) -> List[Tuple[int, int]]:
    """
    Split the text contained in the slice by `glue` and return the respective new slices.
    """
    pos = text.find(glue, slice[0])
    starts = [slice[0]]
    ends = []
    while pos >= 0 and pos + len(glue) <= slice[1]:
        ends.append(pos)
        starts.append(pos + len(glue))
        pos = text.find(glue, pos + 1)

    ends.append(slice[1])
    return list(zip(starts, ends))


def serialize_labeled_span(
    annotation: LabeledSpan, doc: Document, create_id_if_not_available: bool = True
) -> str:
    # We have to remove newline characters from the annotations because this will cause
    # problems for the brat annotation file. So, we create fragments around newlines.
    slices = split_span_annotation(
        text=doc.text, slice=(annotation.start, annotation.end), glue=GLUE_TEXT
    )
    slices_serialized = ";".join([f"{start} {end}" for start, end in slices])
    _text = GLUE_BRAT.join([doc.text[start:end] for start, end in slices])

    serialized_annotation = f"{annotation.label} {slices_serialized}"
    # construct id based on text and annotation
    if annotation.metadata.get("id", None) is None and create_id_if_not_available:
        text_hash = hash_string(doc.text)
        annotation.metadata["id"] = hash_string(
            f"{serialized_annotation} {text_hash}", max_length=8
        )
    return f"T{annotation.metadata['id']}\t{serialized_annotation}\t{_text}\n"


def serialize_binary_relation(
    annotation: BinaryRelation,
    doc: Document,
    head_argument_name: str,
    tail_argument_name: str,
    create_id_if_not_available: bool = True,
) -> str:
    _head_id = annotation.head.metadata["id"]
    _tail_id = annotation.tail.metadata["id"]
    serialized_annotation = (
        f"{annotation.label} {head_argument_name}:T{_head_id} {tail_argument_name}:T{_tail_id}"
    )
    if "id" not in annotation.metadata and create_id_if_not_available:
        annotation.metadata["id"] = hash_string(serialized_annotation, max_length=8)
    return f"R{annotation.metadata['id']}\t{serialized_annotation}\n"


def _write_brat(doc_id: str, text: str, serialized_annotations: List[str], path: str):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, f"{doc_id}.txt"), "w") as txt_file:
        txt_file.write(text)
    with open(os.path.join(path, f"{doc_id}.ann"), "w") as ann_file:
        ann_file.writelines(serialized_annotations)


def convert_document_to_brat(
    doc: Document,
    head_argument_name: str = DEFAULT_HEAD_ARGUMENT_NAME,
    tail_argument_name: str = DEFAULT_TAIL_ARGUMENT_NAME,
    span_annotation_names: Optional[List[str]] = None,
    relation_annotation_names: Optional[List[str]] = None,
    **kwargs,
) -> Tuple[Optional[str], str, List[str]]:
    serialized_annotations: Dict[Annotation, str] = {}
    if span_annotation_names is None:
        span_annotation_names = [DEFAULT_SPAN_ANNOTATION_NAME]
    if relation_annotation_names is None:
        relation_annotation_names = [DEFAULT_RELATION_ANNOTATION_NAME]
    for entity_annotation_name in span_annotation_names:
        if doc.annotations.spans.has_layer(entity_annotation_name):
            for span_ann in doc.annotations.spans[entity_annotation_name]:
                serialized_annotations[span_ann] = serialize_labeled_span(span_ann, doc, **kwargs)
    for relation_annotation_name in relation_annotation_names:
        if doc.annotations.binary_relations.has_layer(relation_annotation_name):
            for rel_ann in doc.annotations.binary_relations[relation_annotation_name]:
                serialized_annotations[rel_ann] = serialize_binary_relation(
                    rel_ann,
                    doc,
                    head_argument_name=head_argument_name,
                    tail_argument_name=tail_argument_name,
                    **kwargs,
                )
    for layer_type, name, annots in doc.annotations.typed_named_layers:
        not_serialized = [ann for ann in annots if ann not in serialized_annotations]
        if len(not_serialized) > 0:
            logger.warning(
                f"{len(not_serialized)} annotations with name '{name}' not serialized to BRAT"
            )

    return doc.id, doc.text, list(serialized_annotations.values())


def _convert_docs(docs: List[Document], **kwargs) -> List[Tuple[Optional[str], str, List[str]]]:
    return [convert_document_to_brat(doc=doc, **kwargs) for doc in docs]


def serialize_brat(
    docs: Dict[str, List[Document]],
    path: str = None,
    create_id_if_not_available: bool = True,
    **kwargs,
) -> Optional[Dict[str, List[Tuple[Optional[str], str, List[str]]]]]:
    res = {}
    for split, _docs in docs.items():
        res[split] = _convert_docs(
            _docs, create_id_if_not_available=create_id_if_not_available, **kwargs
        )
        if path is not None:
            _path = os.path.join(path, split)
            logger.info(
                f"serialize {len(res[split])} documents to BRAT format to directory: {_path}"
            )
            for doc_id, text, serialized_annotations in res[split]:
                if doc_id is None:
                    if create_id_if_not_available:
                        doc_id = hash_string(text, max_length=8)
                    else:
                        raise ValueError(
                            "if create_id_if_not_available=False, the document id has to be specified "
                            "to write the annotated document as brat file"
                        )
                _write_brat(
                    doc_id=doc_id,
                    text=text,
                    serialized_annotations=serialized_annotations,
                    path=_path,
                )

    if path is None:
        return res
    else:
        return None
