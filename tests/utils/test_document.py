from pie_documents.annotations import LabeledSpan
from pie_documents.documents import TextDocumentWithLabeledSpans

from pytorch_ie.utils.document import merge_annotations_from_documents


def test_document_merge_annotations():
    base_doc = TextDocumentWithLabeledSpans(id="doc1", text="This is a test.")
    # add annotations
    base_doc.labeled_spans.append(LabeledSpan(start=0, end=4, label="label1", score=1.0))
    base_doc.labeled_spans.append(LabeledSpan(start=5, end=7, label="label2", score=1.0))

    input1 = base_doc.copy()
    # add predictions
    input1.labeled_spans.predictions.append(LabeledSpan(start=0, end=4, label="label1", score=0.9))
    input1.labeled_spans.predictions.append(LabeledSpan(start=5, end=7, label="label2", score=0.7))

    input2 = base_doc.copy()
    # add predictions
    input2.labeled_spans.predictions.append(LabeledSpan(start=0, end=4, label="label1", score=0.8))
    input2.labeled_spans.predictions.append(LabeledSpan(start=5, end=7, label="label2", score=0.7))
    input2.labeled_spans.predictions.append(LabeledSpan(start=5, end=7, label="label3", score=0.6))

    documents = {
        "doc1": input1,
        "doc2": input2,
    }
    result = merge_annotations_from_documents(
        documents,
        metadata_key_source_annotations="annotations_source",
        metadata_key_source_predictions="predictions_source",
    )
    assert result.id == "doc1"
    assert set(result.labeled_spans) == set(base_doc.labeled_spans)
    assert len(result.labeled_spans) == len(base_doc.labeled_spans) == 2
    assert result.labeled_spans.predictions.resolve() == [
        ("label1", "This"),
        ("label2", "is"),
        ("label3", "is"),
    ]
    annotations_with_sources = [
        (ann.copy(), sources)
        for ann, sources in zip(
            result.labeled_spans, result.metadata["annotations_source"]["labeled_spans"]
        )
    ]
    assert annotations_with_sources == [
        (LabeledSpan(start=0, end=4, label="label1", score=1.0), ["doc1", "doc2"]),
        (LabeledSpan(start=5, end=7, label="label2", score=1.0), ["doc1", "doc2"]),
    ]
    predictions_with_scores = [
        (ann.copy(), sources)
        for ann, sources in zip(
            result.labeled_spans.predictions,
            result.metadata["predictions_source"]["labeled_spans"],
        )
    ]
    assert predictions_with_scores == [
        (LabeledSpan(start=0, end=4, label="label1", score=0.9), ["doc1"]),
        (LabeledSpan(start=5, end=7, label="label2", score=0.7), ["doc1", "doc2"]),
        (LabeledSpan(start=5, end=7, label="label3", score=0.6), ["doc2"]),
    ]
