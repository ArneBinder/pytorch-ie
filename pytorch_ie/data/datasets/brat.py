from typing import Any, Dict, Iterable, List, Optional, Union

from datasets import DatasetDict, load_dataset

from pytorch_ie import Document
from pytorch_ie.data.datasets import HF_DATASETS_ROOT
from pytorch_ie.data.document import BinaryRelation, LabeledMultiSpan, LabeledSpan


def dl_to_ld(dl):
    return [dict(zip(dl, t)) for t in zip(*dl.values())]


# not used
def ld_to_dl(ld):
    return {k: [dic[k] for dic in ld] for k in ld[0]}


def convert_brat_to_document(
    brat_doc: Dict[str, Any],
    head_argument_name: str = "Arg1",
    tail_argument_name: str = "Arg2",
    convert_multi_spans: bool = True,
) -> Document:

    doc = Document(text=brat_doc["context"], doc_id=brat_doc["file_name"])

    # add spans
    span_id_mapping = {}
    for brat_span in dl_to_ld(brat_doc["spans"]):
        locations = dl_to_ld(brat_span["locations"])
        label = brat_span["type"]
        metadata = {"text": brat_span["text"]}
        span: LabeledMultiSpan
        if convert_multi_spans:
            if len(locations) > 1:
                added_fragments = [
                    brat_doc["context"][locations[i]["end"] : locations[i + 1]["start"]]
                    for i in range(len(locations) - 1)
                ]
                print(
                    f"WARNING: convert span with several slices to LabeledSpan! added text fragments: "
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
        brat_args = {arg["type"]: arg["target"] for arg in dl_to_ld(brat_relation["arguments"])}
        head = span_id_mapping[brat_args[head_argument_name]]
        tail = span_id_mapping[brat_args[tail_argument_name]]
        assert isinstance(head, LabeledSpan) and isinstance(tail, LabeledSpan), (
            f"BinaryRelation does only except head and tail of type `{LabeledSpan.__name__}`, "
            f"but they have types `{type(head)}` and `{type(tail)}`."
        )
        relation = BinaryRelation(label=brat_relation["type"], head=head, tail=tail)
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
    **kwargs,
) -> Union[List[Document], Dict[str, List[Document]]]:
    # This will create a DatasetDict with a single split "train"
    path = str(HF_DATASETS_ROOT / "brat.py")
    data = load_dataset(path=path, **kwargs)
    assert isinstance(data, DatasetDict), (
        f"`load_dataset` for `path={path}` should return a DatasetDict (at least containing a single 'train' split), "
        f"but the result is of type: {type(data)}"
    )

    conversion_kwargs = conversion_kwargs or {}
    return {k: _convert_brats(brat_docs=v, **conversion_kwargs) for k, v in data.items()}
