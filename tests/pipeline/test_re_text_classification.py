from dataclasses import dataclass
from importlib.metadata import version

import pytest
import torch
from packaging.version import Version
from pie_core import AnnotationLayer, annotation_field

from pytorch_ie import AutoPipeline
from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.documents import TextDocument
from pytorch_ie.models import TransformerTextClassificationModel
from pytorch_ie.pipeline import Pipeline
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
    # Below scores for torch < 2.6 were obtained with:
    #  - torch==2.3.0, pytorch-lightning==2.2.5, and transformers==4.41.1
    # The scores for torch >= 2.6 were obtained with: TODO update this comment
    #  - torch==2.7.1, pytorch-lightning==2.5.2, and transformers==4.48.3.
    if not half_precision_model and not half_precision_ops:
        assert scores == pytest.approx(
            [0.5339038372039795, 0.3984701931476593, 0.5520647764205933], abs=1e-6
        )
        assert half_precision_warning not in caplog.messages
    elif not half_precision_model and half_precision_ops:
        if Version(version("torch")) < Version("2.6"):
            assert scores == pytest.approx([0.53125, 0.39453125, 0.5546875], abs=1e-6)
        else:
            assert scores == pytest.approx([0.53125, 0.396484375, 0.55078125], abs=1e-6)
        assert half_precision_warning not in caplog.messages
    elif half_precision_model and not half_precision_ops:
        # using half_precision_model on cpu results in using dtype=torch.bfloat16 which has only
        # 8 significant precision bits, so we use 10e-3 as absolute tolerance
        if Version(version("torch")) < Version("2.6"):
            assert scores == pytest.approx([0.53515625, 0.400390625, 0.5546875], abs=1e-3)
        else:
            assert scores == pytest.approx([0.53125, 0.412109375, 0.55859375], abs=1e-3)
        assert half_precision_warning not in caplog.messages
    else:
        # just check that we got the warning about half precision ops in combination with half precision model
        assert half_precision_warning in caplog.messages
