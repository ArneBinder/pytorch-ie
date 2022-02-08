from typing import List

import pytest

from pytorch_ie import Document, Pipeline
from pytorch_ie.data import LabeledSpan
from pytorch_ie.models import TransformerSpanClassificationModel
from pytorch_ie.taskmodules import TransformerSpanClassificationTaskModule


@pytest.mark.slow
def test_ner_span_classification():
    model_name_or_path = "pie/example-ner-spanclf-conll03"
    ner_taskmodule = TransformerSpanClassificationTaskModule.from_pretrained(model_name_or_path)
    ner_model = TransformerSpanClassificationModel.from_pretrained(model_name_or_path)

    ner_pipeline = Pipeline(model=ner_model, taskmodule=ner_taskmodule, device=-1)

    document = Document(
        "“Making a super tasty alt-chicken wing is only half of it,” said Po Bronson, general partner at SOSV and managing director of IndieBio."
    )

    ner_pipeline(document, predict_field="entities", batch_size=2)
    entities: List[LabeledSpan] = document.predictions("entities")  # typing: ignore
    assert len(entities) == 3
    entities_sorted = sorted(entities, key=lambda entity: (entity.start + entity.end) / 2)

    ent0 = entities_sorted[0]
    assert ent0.label_single == "PER"
    assert ent0.score == 1.0
    assert ent0.slices == [(65, 75)]

    ent1 = entities_sorted[1]
    assert ent1.label_single == "ORG"
    assert ent1.score == 1.0
    assert ent1.slices == [(96, 100)]

    ent2 = entities_sorted[2]
    assert ent2.label_single == "ORG"
    assert ent2.score == 1.0
    assert ent2.slices == [(126, 134)]
