import copy
import re
from collections import Counter
from typing import Optional, List, Dict, Iterator

from datasets import Dataset

from pytorch_ie import Document
from pytorch_ie.data import LabeledSpan


def has_overlap(span1: LabeledSpan, span2: LabeledSpan) -> bool:
    return span1.start <= span2.start < span1.end or span1.start < span2.end <= span1.end


def get_sections(
    doc: Document,
    section_pattern: str = r"<([^>/]+)>.*</\1>",
    section_label_whitelist: Optional[List[str]] = None,
) -> Iterator[LabeledSpan]:
    """
    Spans are created starting with the first matching entry end ending with the start of the next matching one or
    the end of the document. Entries with xml node names that are not in section_label_whitelist, if it is set, will
    not be considered as match, i.e. their content will be added to the previous matching entry.
    """
    previous_start = previous_header_type = previous_header_text = None
    for match in re.finditer(pattern=section_pattern, string=doc.text):
        header_type = doc.text[match.start(1): match.end(1)]
        if section_label_whitelist is None or header_type in section_label_whitelist:
            if previous_start is not None:
                end = match.start()
                span = LabeledSpan(
                    start=previous_start, end=end, label=previous_header_type,
                    metadata={"text": doc.text[previous_start:end], "header_text": previous_header_text}
                )
                yield span

            previous_header_text = doc.text[match.start(): match.end()]
            previous_header_type = header_type
            previous_start = match.start()

    if previous_start is not None:
        end = len(doc.text)
        span = LabeledSpan(
            start=previous_start, end=end, label=previous_header_type,
            metadata={"text": doc.text[previous_start:end], "header_text": previous_header_text}
        )
        yield span


def annotate_with_sections(
    doc: Document, section_label_whitelist: Optional[List[str]] = None, section_label_stats: Optional[Counter] = None
):
    sections_for_doc = []
    for section in get_sections(doc, section_label_whitelist=section_label_whitelist):
        for s in sections_for_doc:
            if has_overlap(s, section):
                print(f'WARNING: overlap: {section} with {s}')
        if section_label_stats is not None:
            section_label_stats[section.label] += 1
        doc.add_annotation(name="sections", annotation=section)
        sections_for_doc.append(section)
    return sections_for_doc


def annotate_dataset_with_sections(
    dataset: Dataset,
    section_label_whitelist: Optional[List[str]] = ["Title", "Abstract", "H1"],
    inplace: bool = True,
):
    if not inplace:
        dataset = copy.deepcopy(dataset)
    section_label_stats = Counter()
    for doc in dataset:
        annotate_with_sections(
            doc=doc, section_label_whitelist=section_label_whitelist, section_label_stats=section_label_stats
        )
    print(f'identified sections: {section_label_stats}')
    if not inplace:
        return dataset

