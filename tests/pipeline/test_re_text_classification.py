from dataclasses import dataclass

import pytest
from pie_core import AnnotationLayer, annotation_field

from pytorch_ie import AutoPipeline
from pytorch_ie.annotations import BinaryRelation, LabeledSpan
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
@pytest.mark.parametrize("half_precision_model", [False, True])
@pytest.mark.parametrize("half_precision_ops", [False, True])
def test_re_text_classification(use_auto, half_precision_model, half_precision_ops):

    # set up the pipeline
    model_name_or_path = "pie/example-re-textclf-tacred"
    if use_auto:
        pipeline = AutoPipeline.from_pretrained(
            model_name_or_path,
            taskmodule_kwargs={"create_relation_candidates": True},
            half_precision_model=half_precision_model,
        )
    else:
        re_taskmodule = TransformerRETextClassificationTaskModule.from_pretrained(
            model_name_or_path,
            create_relation_candidates=True,
        )
        re_model = TransformerTextClassificationModel.from_pretrained(model_name_or_path)
        pipeline = Pipeline(
            model=re_model,
            taskmodule=re_taskmodule,
            device=-1,
            half_precision_model=half_precision_model,
        )
    assert pipeline.taskmodule.is_from_pretrained
    assert pipeline.model.is_from_pretrained

    # create a document with entities
    document = ExampleDocument(
        "“Making a super tasty alt-chicken wing is only half of it,” said Po Bronson, general partner "
        "at SOSV and managing director of IndieBio."
    )
    document.entities.append(LabeledSpan(start=65, end=75, label="PER"))
    document.entities.append(LabeledSpan(start=96, end=100, label="ORG"))
    document.entities.append(LabeledSpan(start=126, end=134, label="ORG"))

    # predict relations
    pipeline(document, batch_size=2, half_precision_ops=half_precision_ops)

    # sort to get deterministic order
    sorted_relations = sorted(document.relations.predictions)

    # check the relations and their scores
    assert [ann.resolve() for ann in sorted_relations] == [
        ("per:employee_of", (("PER", "Po Bronson"), ("ORG", "IndieBio"))),
        ("org:top_members/employees", (("ORG", "SOSV"), ("PER", "Po Bronson"))),
        ("org:top_members/employees", (("ORG", "IndieBio"), ("PER", "Po Bronson"))),
    ]
    scores = [rel.score for rel in sorted_relations]
    # Note: The scores are quite low, because the model is trained with the old version for the taskmodule,
    # so the argument markers are not correct.
    assert scores == pytest.approx([0.5339038, 0.3984702, 0.5520648], abs=1e-2)
