from typing import List

import pytest

from pytorch_ie import Document, Pipeline
from pytorch_ie.data import BinaryRelation, LabeledSpan
from pytorch_ie.models import TransformerTextClassificationModel
from pytorch_ie.taskmodules import TransformerRETextClassificationTaskModule


@pytest.mark.slow
def test_re_text_classification():
    model_name_or_path = "pie/example-re-textclf-tacred"
    re_taskmodule = TransformerRETextClassificationTaskModule.from_pretrained(model_name_or_path)
    re_model = TransformerTextClassificationModel.from_pretrained(model_name_or_path)

    pipeline = Pipeline(model=re_model, taskmodule=re_taskmodule, device=-1)

    document = Document(
        "“Making a super tasty alt-chicken wing is only half of it,” said Po Bronson, general partner at SOSV and managing director of IndieBio."
    )

    for start, end, label in [(65, 75, "PER"), (96, 100, "ORG"), (126, 134, "ORG")]:
        document.add_annotation("entities", LabeledSpan(start, end, label))

    pipeline(document, predict_field="relations", batch_size=2)
    relations: List[BinaryRelation] = document.predictions("relations")  # typing: ignore
    assert len(relations) == 4
    sorted_relations = sorted(relations, key=lambda rel: (rel.head.start + rel.tail.start) / 2)

    rel0 = sorted_relations[0]
    assert rel0.label_single == "per:employee_of"
    assert rel0.score == 0.9694293737411499
    assert (rel0.head.start, rel0.head.end) == (65, 75)
    assert (rel0.tail.start, rel0.tail.end) == (96, 100)

    rel1 = sorted_relations[1]
    assert rel1.label_single == "org:top_members/employees"
    assert rel1.score == 0.7182475924491882
    assert (rel1.head.start, rel1.head.end) == (96, 100)
    assert (rel1.tail.start, rel1.tail.end) == (65, 75)

    rel2 = sorted_relations[2]
    assert rel2.label_single == "per:employee_of"
    assert rel2.score == 0.9422258138656616
    assert (rel2.head.start, rel2.head.end) == (65, 75)
    assert (rel2.tail.start, rel2.tail.end) == (126, 134)

    rel3 = sorted_relations[3]
    assert rel3.label_single == "org:top_members/employees"
    assert rel3.score == 0.8561000823974609
    assert (rel3.head.start, rel3.head.end) == (126, 134)
    assert (rel3.tail.start, rel3.tail.end) == (65, 75)
