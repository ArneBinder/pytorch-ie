from dataclasses import dataclass

import pytest
import torch
from pie_core import AnnotationLayer, annotation_field
from pie_documents.annotations import BinaryRelation, LabeledSpan
from pie_documents.documents import TextDocument

from pytorch_ie import PyTorchIEPipeline
from pytorch_ie.models import TransformerTextClassificationModel
from pytorch_ie.taskmodules import TransformerRETextClassificationTaskModule

torch.use_deterministic_algorithms(True)


@dataclass
class ExampleDocument(TextDocument):
    entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationLayer[BinaryRelation] = annotation_field(target="entities")


@pytest.mark.slow
@pytest.mark.parametrize("use_auto", [False, True])
@pytest.mark.parametrize("half_precision_model", [False, True])
@pytest.mark.parametrize("half_precision_ops", [False, True])
def test_re_text_classification(use_auto, half_precision_model, half_precision_ops, caplog):

    # set up the pipeline
    model_name_or_path = "pie/example-re-textclf-tacred"
    if use_auto:
        pipeline = PyTorchIEPipeline.from_pretrained(
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
        pipeline = PyTorchIEPipeline(
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
    with caplog.at_level("WARNING"):
        pipeline(document, batch_size=2, half_precision_ops=half_precision_ops)

    # sort to get deterministic order
    sorted_relations = sorted(document.relations.predictions)

    # check the relations and their scores
    assert [ann.resolve() for ann in sorted_relations] == [
        ("per:employee_of", (("PER", "Po Bronson"), ("ORG", "IndieBio"))),
        ("org:top_members/employees", (("ORG", "SOSV"), ("PER", "Po Bronson"))),
        ("org:top_members/employees", (("ORG", "IndieBio"), ("PER", "Po Bronson"))),
    ]

    half_precision_warning = (
        "Using half precision operations with a model already in half precision. "
        "This is not recommended, as it may lead to unexpected results."
    )

    scores = [rel.score for rel in sorted_relations]
    # General note: The scores are quite low, because the model is trained with the old version
    # for the taskmodule, so the argument markers are not correct.
    # Below scores were obtained with dependencies from poetry.lock on local machine.
    if not half_precision_model and not half_precision_ops:
        # we use low tolerance if no half precision is used
        # (i.e., no autocast on forward pass and model is not cast to half precision)
        assert scores == pytest.approx(
            [0.5339038372039795, 0.3984701931476593, 0.5520647764205933], abs=1e-4
        )
        assert half_precision_warning not in caplog.messages
    elif not half_precision_model and half_precision_ops:
        # set high tolerance for half precision ops (i.e., autocast on forward pass)
        assert scores == pytest.approx([0.53125, 0.39453125, 0.5546875], abs=1e-2)
        assert half_precision_warning not in caplog.messages
    elif half_precision_model and not half_precision_ops:
        # set high tolerance for half precision model (i.e., model cast to half precision)
        assert scores == pytest.approx([0.53515625, 0.400390625, 0.55859375], abs=1e-2)
        assert half_precision_warning not in caplog.messages
    else:
        # just check that we got the warning about half precision ops in combination with half precision model
        assert half_precision_warning in caplog.messages
