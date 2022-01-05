import copy
from collections import Counter
from typing import Optional, List, Iterator, Callable, Iterable, Match

from datasets import Dataset

from pytorch_ie import Document
from pytorch_ie.data import LabeledSpan


def has_overlap(span1: LabeledSpan, span2: LabeledSpan) -> bool:
    return span1.start <= span2.start < span1.end or span1.start < span2.end <= span1.end


def get_partitions_with_matcher(
    doc: Document,
    matcher: Optional[Callable[[str], Iterable[Match]]],  # = lambda text: re.finditer(pattern=r"<([^>/]+)>.*</\1>", string=text)
    label_group_id: Optional[int] = None,  # = 1,
    label_whitelist: Optional[List[str]] = None,
    skip_initial_partition: bool = False,  # = True
) -> Iterator[LabeledSpan]:
    """
    Spans are created starting with the beginning of matching entries end ending with the start of the next matching
    one or the end of the document. Entries with xml node names that are not in label_whitelist, if it is set, will
    not be considered as match, i.e. their content will be added to the previous matching entry.
    If label_group_id is set, the content of the respective match group will be taken as label. Otherwise, it is set to
    None. If the flag skip_initial_partition is enabled, the content before the first match is not added as a partition.
    Note that the initial partition will get None as label since no matched element is available.
    """
    previous_start = previous_label = previous_match_text = None
    if not skip_initial_partition:
        previous_start = 0
    for match in matcher(doc.text):
        if label_group_id is not None:
            label = doc.text[match.start(label_group_id): match.end(label_group_id)]
        else:
            label = None
        if label_whitelist is None or label in label_whitelist:
            if previous_start is not None:
                end = match.start()
                span = LabeledSpan(
                    start=previous_start, end=end, label=previous_label,
                    metadata={"text": doc.text[previous_start:end], "header_text": previous_match_text}
                )
                yield span

            previous_match_text = doc.text[match.start(): match.end()]
            previous_label = label
            previous_start = match.start()

    if previous_start is not None:
        end = len(doc.text)
        span = LabeledSpan(
            start=previous_start, end=end, label=previous_label,
            metadata={"text": doc.text[previous_start:end], "header_text": previous_match_text}
        )
        yield span


def annotate_document_with_partitions(
    doc: Document,
    annotation_name: str,
    label_stats: Optional[Counter] = None,
    **kwargs,
):
    partitions_for_doc = []
    for partition in get_partitions_with_matcher(doc, **kwargs):
        # just a sanity check
        for s in partitions_for_doc:
            if has_overlap(s, partition):
                print(f'WARNING: overlap: {partition} with {s}')
        # just some statistics
        if label_stats is not None:
            label_stats[partition.label] += 1
        doc.add_annotation(name=annotation_name, annotation=partition)
        partitions_for_doc.append(partition)
    return partitions_for_doc


def annotate_dataset_with_partitions(
    dataset: Dataset,
    inplace: bool = True,
    **kwargs
):
    if not inplace:
        dataset = copy.deepcopy(dataset)
    partition_label_stats = Counter()
    for doc in dataset:
        annotate_document_with_partitions(doc=doc, label_stats=partition_label_stats, **kwargs)
    print(f'identified partitions: {partition_label_stats}')
    if not inplace:
        return dataset

