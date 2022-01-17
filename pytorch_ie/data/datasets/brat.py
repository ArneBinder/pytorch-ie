import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

from datasets import DatasetDict, load_dataset

from pytorch_ie import Document
from pytorch_ie.data.datasets import HF_DATASETS_ROOT
from pytorch_ie.data.document import BinaryRelation, LabeledMultiSpan, LabeledSpan

DEFAULT_HEAD_ARGUMENT_NAME: str = "Arg1"
DEFAULT_TAIL_ARGUMENT_NAME: str = "Arg2"

logger = logging.getLogger(__name__)


def dl_to_ld(dl):
    return [dict(zip(dl, t)) for t in zip(*dl.values())]


# not used
def ld_to_dl(ld):
    return {k: [dic[k] for dic in ld] for k in ld[0]}


def convert_brat_to_document(
    brat_doc: Dict[str, Any],
    head_argument_name: str = DEFAULT_HEAD_ARGUMENT_NAME,
    tail_argument_name: str = DEFAULT_TAIL_ARGUMENT_NAME,
    convert_multi_spans: bool = True,
) -> Document:

    doc = Document(text=brat_doc["context"], doc_id=brat_doc["file_name"])

    # add spans
    span_id_mapping = {}
    for brat_span in dl_to_ld(brat_doc["spans"]):
        locations = dl_to_ld(brat_span["locations"])
        label = brat_span["type"]
        metadata = {"text": brat_span["text"], "id": brat_span["id"]}
        span: LabeledMultiSpan
        if convert_multi_spans:
            if len(locations) > 1:
                added_fragments = [
                    brat_doc["context"][locations[i]["end"] : locations[i + 1]["start"]]
                    for i in range(len(locations) - 1)
                ]
                logger.warning(
                    f"convert span with several slices to LabeledSpan! added text fragments: "
                    f"{added_fragments}"
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
        doc.add_annotation(name="entities", annotation=span)

    # add relations
    for brat_relation in dl_to_ld(brat_doc["relations"]):
        metadata = {"id": brat_relation["id"]}
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
        doc.add_annotation(name="relations", annotation=relation)

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


def serialize_labeled_span(annotation: LabeledSpan, doc: Document) -> str:
    _id = annotation.metadata["id"]
    _text = doc.text[annotation.start : annotation.end]
    return f"{_id}\t{annotation.label} {annotation.start} {annotation.end}\t{_text}"


def serialize_binary_relation(
    annotation: BinaryRelation, doc: Document, head_argument_name: str, tail_argument_name: str
) -> str:
    _id = annotation.metadata["id"]
    _head_id = annotation.head.metadata["id"]
    _tail_id = annotation.tail.metadata["id"]
    return f"{_id}\t{annotation.label} {head_argument_name}:{_head_id} {tail_argument_name}:{_tail_id}"


def _write_brat(doc_id: str, text: str, serialized_annotations: List[str], path: str):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, f"{doc_id}.txt"), "w") as txt_file:
        txt_file.write(text)
    with open(os.path.join(path, f"{doc_id}.ann"), "w") as ann_file:
        ann_file.writelines([f"{ann}\n" for ann in serialized_annotations])


def convert_document_to_brat(
    doc: Document,
    head_argument_name=DEFAULT_HEAD_ARGUMENT_NAME,
    tail_argument_name=DEFAULT_TAIL_ARGUMENT_NAME,
) -> Tuple[Optional[str], str, List[str]]:
    serialized_annotations = []
    for name, annots in doc._annotations.items():
        for ann in annots:
            if isinstance(ann, LabeledSpan):
                serialized_annotations.append(serialize_labeled_span(ann, doc))
            elif isinstance(ann, BinaryRelation):
                serialized_annotations.append(
                    serialize_binary_relation(
                        ann,
                        doc,
                        head_argument_name=head_argument_name,
                        tail_argument_name=tail_argument_name,
                    )
                )
            else:
                raise NotImplementedError(
                    f"Serialization to Brat for annotation of type '{type(ann)}' not yet implemented."
                )

    return doc.id, doc.text, serialized_annotations


def _convert_docs(docs: List[Document], **kwargs) -> List[Tuple[Optional[str], str, List[str]]]:
    return [convert_document_to_brat(doc=doc, **kwargs) for doc in docs]


def serialize_brat(
    docs: Dict[str, List[Document]], path: str = None, **kwargs
) -> Optional[Dict[str, List[Tuple[Optional[str], str, List[str]]]]]:
    res = {}
    for split, _docs in docs.items():
        res[split] = _convert_docs(_docs, **kwargs)
        if path is not None:
            _path = os.path.join(path, split)
            logger.info(
                f"serialize {len(res[split])} documents to BRAT format to directory: {_path}"
            )
            for doc_id, text, serialized_annotations in res[split]:
                assert (
                    doc_id is not None
                ), "the document id has to be specified to write the annotated document as brat file"
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
