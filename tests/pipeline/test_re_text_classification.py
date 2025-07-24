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
    # General note: The scores are quite low, because the model is trained with the old version
    # for the taskmodule, so the argument markers are not correct.
    if not half_precision_model and not half_precision_ops:
        assert scores == pytest.approx(
            [0.5339038372039795, 0.3984701931476593, 0.5520647764205933]
        )
    elif not half_precision_model and half_precision_ops:
        assert scores == pytest.approx([0.53125, 0.39453125, 0.5546875])
    elif half_precision_model and not half_precision_ops:
        assert scores == pytest.approx([0.53515625, 0.400390625, 0.55859375])
    else:
        # NOTE: This should not be used, see recommendation from torch.autocast() documentation:
        # "When entering an autocast-enabled region, Tensors may be any type. You should not call
        # half() or bfloat16() on your model(s) or inputs when using autocasting."
        assert scores == pytest.approx([0.53515625, 0.400390625, 0.55859375])
