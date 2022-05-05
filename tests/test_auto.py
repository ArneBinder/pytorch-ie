from dataclasses import dataclass

import pytest

from pytorch_ie.annotations import LabeledSpan
from pytorch_ie.auto import AutoModel, AutoPipeline, AutoTaskModule
from pytorch_ie.core import AnnotationList, TaskModule, annotation_field
from pytorch_ie.documents import TextDocument
from pytorch_ie.models import TransformerSpanClassificationModel
from pytorch_ie.taskmodules import TransformerSpanClassificationTaskModule


@pytest.mark.slow
def test_auto_taskmodule():
    taskmodule = AutoTaskModule.from_pretrained("pie/example-ner-spanclf-conll03")
    assert isinstance(taskmodule, TransformerSpanClassificationTaskModule)
    assert taskmodule.label_to_id == {"O": 0, "MISC": 1, "ORG": 2, "PER": 3, "LOC": 4}


@pytest.mark.slow
def test_auto_taskmodule_full_cycle(tmp_path):
    @TaskModule.register()
    class MyTransformerSpanClassificationTaskModule(TransformerSpanClassificationTaskModule):
        pass

    taskmodule = MyTransformerSpanClassificationTaskModule(
        tokenizer_name_or_path="bert-base-uncased"
    )
    taskmodule.prepare([])
    taskmodule.save_pretrained(save_directory=str(tmp_path))

    taskmodule_loaded = AutoTaskModule.from_pretrained(str(tmp_path))
    assert isinstance(taskmodule_loaded, MyTransformerSpanClassificationTaskModule)


@pytest.mark.slow
def test_auto_taskmodule_full_cycle_with_name(tmp_path):
    @TaskModule.register(name="mytaskmodule")
    class MyTransformerSpanClassificationTaskModule(TransformerSpanClassificationTaskModule):
        pass

    taskmodule = MyTransformerSpanClassificationTaskModule(
        tokenizer_name_or_path="bert-base-uncased"
    )
    taskmodule.prepare([])
    taskmodule.save_pretrained(save_directory=str(tmp_path))

    taskmodule_loaded = AutoTaskModule.from_pretrained(str(tmp_path))
    assert isinstance(taskmodule_loaded, MyTransformerSpanClassificationTaskModule)


@pytest.mark.slow
def test_auto_model():
    model = AutoModel.from_pretrained("pie/example-ner-spanclf-conll03")
    assert isinstance(model, TransformerSpanClassificationModel)


@pytest.mark.slow
def test_auto_pipeline():
    @dataclass
    class ExampleDocument(TextDocument):
        entities: AnnotationList[LabeledSpan] = annotation_field(target="text")

    pipeline = AutoPipeline.from_pretrained("pie/example-ner-spanclf-conll03")

    document = ExampleDocument(
        "“Making a super tasty alt-chicken wing is only half of it,” said Po Bronson, general partner at SOSV and managing director of IndieBio."
    )

    pipeline(document, predict_field="entities", num_workers=0)

    entities = document.entities.predictions
    assert len(entities) == 3
