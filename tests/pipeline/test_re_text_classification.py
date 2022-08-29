from dataclasses import dataclass
from typing import Sequence

import pytest

from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.documents import TextDocument
from pytorch_ie.models import TransformerTextClassificationModel
from pytorch_ie.pipeline import Pipeline
from pytorch_ie.taskmodules import TransformerRETextClassificationTaskModule


@dataclass
class ExampleDocument(TextDocument):
    entities: AnnotationList[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")


@pytest.mark.slow
def test_re_text_classification():
    model_name_or_path = "pie/example-re-textclf-tacred"
    re_taskmodule = TransformerRETextClassificationTaskModule.from_pretrained(model_name_or_path)
    assert re_taskmodule.is_from_pretrained
    re_model = TransformerTextClassificationModel.from_pretrained(model_name_or_path)
    assert re_model.is_from_pretrained

    pipeline = Pipeline(model=re_model, taskmodule=re_taskmodule, device=-1)

    document = ExampleDocument(
        "“Making a super tasty alt-chicken wing is only half of it,” said Po Bronson, general partner at SOSV and managing director of IndieBio."
    )

    for start, end, label in [(65, 75, "PER"), (96, 100, "ORG"), (126, 134, "ORG")]:
        document.entities.append(LabeledSpan(start=start, end=end, label=label))

    pipeline(document, predict_field="relations", batch_size=2)
    relations: Sequence[BinaryRelation] = document["relations"].predictions
    assert len(relations) == 4

    sorted_relations = sorted(relations, key=lambda rel: (rel.head.start + rel.tail.start) / 2)

    relation0 = sorted_relations[0]
    assert relation0.label == "per:employee_of"
    assert relation0.score == pytest.approx(0.96, abs=1e-2)
    assert (relation0.head.start, relation0.head.end) == (65, 75)
    assert (relation0.tail.start, relation0.tail.end) == (96, 100)

    relation1 = sorted_relations[1]
    assert relation1.label == "org:top_members/employees"
    assert relation1.score == pytest.approx(0.71, abs=1e-2)
    assert (relation1.head.start, relation1.head.end) == (96, 100)
    assert (relation1.tail.start, relation1.tail.end) == (65, 75)

    relation2 = sorted_relations[2]
    assert relation2.label == "per:employee_of"
    assert relation2.score == pytest.approx(0.94, abs=1e-2)
    assert (relation2.head.start, relation2.head.end) == (65, 75)
    assert (relation2.tail.start, relation2.tail.end) == (126, 134)

    relation3 = sorted_relations[3]
    assert relation3.label == "org:top_members/employees"
    assert relation3.score == pytest.approx(0.85, abs=1e-2)
    assert (relation3.head.start, relation3.head.end) == (126, 134)
    assert (relation3.tail.start, relation3.tail.end) == (65, 75)
