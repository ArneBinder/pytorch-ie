from dataclasses import dataclass

import pytest
from pie_core import AnnotationLayer, annotation_field

from pytorch_ie import PyTorchIEPipeline
from pytorch_ie.annotations import LabeledSpan
from pytorch_ie.documents import TextDocument
from pytorch_ie.models import TransformerSpanClassificationModel
from pytorch_ie.taskmodules import TransformerSpanClassificationTaskModule


@dataclass
class ExampleDocument(TextDocument):
    entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")


@pytest.mark.slow
@pytest.mark.parametrize("fast_dev_run", [False, True])
@pytest.mark.parametrize("use_auto", [False, True])
def test_ner_span_classification(fast_dev_run, use_auto):
    model_name_or_path = "pie/example-ner-spanclf-conll03"
    if use_auto:
        ner_pipeline = PyTorchIEPipeline.from_pretrained(
            model_name_or_path, device=-1, fast_dev_run=fast_dev_run
        )
    else:
        ner_taskmodule = TransformerSpanClassificationTaskModule.from_pretrained(
            model_name_or_path
        )
        ner_model = TransformerSpanClassificationModel.from_pretrained(model_name_or_path)
        ner_pipeline = PyTorchIEPipeline(
            model=ner_model, taskmodule=ner_taskmodule, device=-1, fast_dev_run=fast_dev_run
        )
    assert ner_pipeline.taskmodule.is_from_pretrained
    assert ner_pipeline.model.is_from_pretrained

    text = "“Making a super tasty alt-chicken wing is only half of it,” said Po Bronson, general partner at SOSV and managing director of IndieBio."
    documents = [ExampleDocument(text), ExampleDocument(text), ExampleDocument(text)]
    ner_pipeline(documents, batch_size=2)

    for i, document in enumerate(documents):
        entities = document.entities.predictions
        if i > 1 and fast_dev_run:
            assert len(entities) == 0
            continue
        assert len(entities) == 3
        entities_sorted = sorted(entities, key=lambda entity: (entity.start + entity.end) / 2)

        entity1 = entities_sorted[0]
        assert entity1.label == "PER"
        assert entity1.score == pytest.approx(0.98, abs=1e-2)
        assert (entity1.start, entity1.end) == (65, 75)

        entity2 = entities_sorted[1]
        assert entity2.label == "ORG"
        assert entity2.score == pytest.approx(0.96, abs=1e-2)
        assert (entity2.start, entity2.end) == (96, 100)

        entity3 = entities_sorted[2]
        assert entity3.label == "ORG"
        assert entity3.score == pytest.approx(0.95, abs=1e-2)
        assert (entity3.start, entity3.end) == (126, 134)
