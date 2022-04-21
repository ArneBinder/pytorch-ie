from dataclasses import dataclass

import pytest

from pytorch_ie import AnnotationList, LabeledSpan, Pipeline, TextDocument, annotation_field
from pytorch_ie.models import TransformerSpanClassificationModel
from pytorch_ie.taskmodules import TransformerSpanClassificationTaskModule


@dataclass
class ExampleDocument(TextDocument):
    entities: AnnotationList[LabeledSpan] = annotation_field(target="text")


@pytest.mark.slow
def test_ner_span_classification():
    model_name_or_path = "pie/example-ner-spanclf-conll03"
    ner_taskmodule = TransformerSpanClassificationTaskModule.from_pretrained(model_name_or_path)
    ner_model = TransformerSpanClassificationModel.from_pretrained(model_name_or_path)

    ner_pipeline = Pipeline(model=ner_model, taskmodule=ner_taskmodule, device=-1)

    document0 = ExampleDocument(
        "“Making a super tasty alt-chicken wing is only half of it,” said Po Bronson, general partner at SOSV and managing director of IndieBio."
    )
    document1 = ExampleDocument(
        "“Making a super tasty alt-chicken wing is only half of it,” said Po Bronson, general partner at SOSV and managing director of IndieBio."
    )
    documents = [document0, document1]
    ner_pipeline(documents, predict_field="entities", batch_size=2)

    for document in documents:
        entities = document.entities.predictions
        assert len(entities) == 3
        entities_sorted = sorted(entities, key=lambda entity: (entity.start + entity.end) / 2)

        entity1 = entities_sorted[0]
        assert entity1.label == "PER"
        assert entity1.score == pytest.approx(1.0)
        assert (entity1.start, entity1.end) == (65, 75)

        entity2 = entities_sorted[1]
        assert entity2.label == "ORG"
        assert entity2.score == pytest.approx(1.0)
        assert (entity2.start, entity2.end) == (96, 100)

        entity3 = entities_sorted[2]
        assert entity3.label == "ORG"
        assert entity3.score == pytest.approx(1.0)
        assert (entity3.start, entity3.end) == (126, 134)
