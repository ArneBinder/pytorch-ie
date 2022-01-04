from typing import List, Dict, Optional, Any

from datasets import load_dataset, Dataset

from pytorch_ie import Document
from pytorch_ie.data.document import LabeledMultiSpan, BinaryRelation, LabeledSpan


def dl_to_ld(dl):
    return [dict(zip(dl, t)) for t in zip(*dl.values())]


# not used
def ld_to_dl(ld):
    return {k: [dic[k] for dic in ld] for k in ld[0]}


def convert_brat_dataset_to_documents(
    dataset: Dataset,
    head_argument_name: str = "Arg1",
    tail_argument_name: str = "Arg2",
    convert_multi_spans: bool = True
) -> List[Document]:
    docs = []
    for brat_doc in dataset:
        doc = Document(text=brat_doc["context"], doc_id=brat_doc["file_name"])

        # add spans
        span_id_mapping = {}
        for brat_span in dl_to_ld(brat_doc["spans"]):
            locations = dl_to_ld(brat_span["locations"])
            label = brat_span["type"]
            metadata = {"text": brat_span["text"]}
            if convert_multi_spans:
                if len(locations) > 1:
                    added_fragments = [
                        brat_doc["context"][locations[i]["end"]:locations[i+1]["start"]]
                        for i in range(len(locations)-1)
                    ]
                    print(f"WARNING: convert span with several slices to LabeledSpan! added text fragments: "
                          f"{added_fragments}")
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
            assert brat_span["id"] not in span_id_mapping, f'brat span id "{brat_span["id"]}" already exists'
            span_id_mapping[brat_span["id"]] = span
            doc.add_annotation(name="entities", annotation=span)

        # add relations
        for brat_relation in dl_to_ld(brat_doc["relations"]):
            brat_args = {arg["type"]: arg["target"] for arg in dl_to_ld(brat_relation["arguments"])}
            head = span_id_mapping[brat_args[head_argument_name]]
            tail = span_id_mapping[brat_args[tail_argument_name]]
            relation = BinaryRelation(label=brat_relation["type"], head=head, tail=tail)
            doc.add_annotation(name="relations", annotation=relation)

        # add events -> not yet implement
        # add equivalence_relations -> not yet implement
        # add attributions -> not yet implement
        # add normalizations -> not yet implement
        # add notes -> not yet implement

        docs.append(doc)
    return docs


def load_brat(
    train_test_split: Optional[Dict[str, Any]] = None, conversion_kwargs: Dict[str, Any] = {}, **kwargs
) -> Dict[str, List[Document]]:
    # This will create a DatasetDict with a single split "train"
    data = load_dataset(path='./pytorch_ie/data/datasets/hf_datasets/brat.py', **kwargs)
    if train_test_split is not None:
        data = data["train"].train_test_split(**train_test_split)

    return {split: convert_brat_dataset_to_documents(dataset, **conversion_kwargs) for split, dataset in data.items()}
