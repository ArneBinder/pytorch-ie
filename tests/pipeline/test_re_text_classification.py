from dataclasses import dataclass
from typing import Sequence

import pytest

from pytorch_ie import AutoPipeline
from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.core import AnnotationLayer, annotation_field
from pytorch_ie.documents import TextDocument
from pytorch_ie.models import TransformerTextClassificationModel
from pytorch_ie.pipeline import Pipeline
from pytorch_ie.taskmodules import TransformerRETextClassificationTaskModule


@dataclass
class ExampleDocument(TextDocument):
    entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationLayer[BinaryRelation] = annotation_field(target="entities")


@pytest.mark.slow
@pytest.mark.parametrize("use_auto", [False, True])
def test_re_text_classification(use_auto):
    model_name_or_path = "pie/example-re-textclf-tacred"
    if use_auto:
        pipeline = AutoPipeline.from_pretrained(
            model_name_or_path, taskmodule_kwargs={"create_relation_candidates": True}
        )
    else:
        re_taskmodule = TransformerRETextClassificationTaskModule.from_pretrained(
            model_name_or_path,
            create_relation_candidates=True,
        )
        re_model = TransformerTextClassificationModel.from_pretrained(model_name_or_path)
        pipeline = Pipeline(model=re_model, taskmodule=re_taskmodule, device=-1)
    assert pipeline.taskmodule.is_from_pretrained
    assert pipeline.model.is_from_pretrained

    document = ExampleDocument(
        "“Making a super tasty alt-chicken wing is only half of it,” said Po Bronson, general partner "
        "at SOSV and managing director of IndieBio."
    )

    for start, end, label in [(65, 75, "PER"), (96, 100, "ORG"), (126, 134, "ORG")]:
        document.entities.append(LabeledSpan(start=start, end=end, label=label))

    pipeline(document, batch_size=2)
    relations: Sequence[BinaryRelation] = document["relations"].predictions
    assert len(relations) == 3

    rels = sorted(relations, key=lambda rel: (rel.head.start + rel.tail.start) / 2)

    # Note: The scores are quite low, because the model is trained with the old version for the taskmodule,
    # so the argument markers are not correct.
    assert (str(rels[0].head), rels[0].label, str(rels[0].tail)) == (
        "SOSV",
        "org:top_members/employees",
        "Po Bronson",
    )
    assert rels[0].score == pytest.approx(0.398, abs=1e-2)

    assert (str(rels[1].head), rels[1].label, str(rels[1].tail)) == (
        "Po Bronson",
        "per:employee_of",
        "IndieBio",
    )
    assert rels[1].score == pytest.approx(0.534, abs=1e-2)

    assert (str(rels[2].head), rels[2].label, str(rels[2].tail)) == (
        "IndieBio",
        "org:top_members/employees",
        "Po Bronson",
    )
    assert rels[2].score == pytest.approx(0.552, abs=1e-2)
