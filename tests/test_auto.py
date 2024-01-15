from dataclasses import dataclass

import pytest

from pytorch_ie import PyTorchIEModel
from pytorch_ie.annotations import LabeledSpan
from pytorch_ie.auto import AutoModel, AutoPipeline, AutoTaskModule
from pytorch_ie.core import AnnotationLayer, TaskModule, annotation_field
from pytorch_ie.documents import TextDocument
from pytorch_ie.models import TransformerSpanClassificationModel
from pytorch_ie.taskmodules import TransformerSpanClassificationTaskModule


@TaskModule.register()
class MyTransformerSpanClassificationTaskModule(TransformerSpanClassificationTaskModule):
    pass


@PyTorchIEModel.register()
class MyTransformerSpanClassificationModel(TransformerSpanClassificationModel):
    pass


@pytest.mark.slow
def test_auto_taskmodule():
    taskmodule = AutoTaskModule.from_pretrained("pie/example-ner-spanclf-conll03")
    assert isinstance(taskmodule, TransformerSpanClassificationTaskModule)
    assert taskmodule.label_to_id == {"O": 0, "MISC": 1, "ORG": 2, "PER": 3, "LOC": 4}
    assert taskmodule.is_prepared


@pytest.mark.slow
def test_auto_taskmodule_full_cycle(tmp_path):

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
    config = taskmodule._config()
    assert config["taskmodule_type"] == "mytaskmodule"
    taskmodule.prepare([])
    taskmodule.save_pretrained(save_directory=str(tmp_path))

    taskmodule_loaded = AutoTaskModule.from_pretrained(str(tmp_path))
    assert isinstance(taskmodule_loaded, MyTransformerSpanClassificationTaskModule)


def test_auto_taskmodule_from_config():

    config = {
        "taskmodule_type": "MyTransformerSpanClassificationTaskModule",
        "tokenizer_name_or_path": "bert-base-uncased",
    }
    taskmodule = AutoTaskModule.from_config(config)
    assert isinstance(taskmodule, MyTransformerSpanClassificationTaskModule)
    assert taskmodule.is_prepared


@pytest.mark.slow
def test_auto_model():
    model = AutoModel.from_pretrained("pie/example-ner-spanclf-conll03")
    assert isinstance(model, TransformerSpanClassificationModel)


def test_auto_model_from_config():
    config = {
        "model_type": "MyTransformerSpanClassificationModel",
        "model_name_or_path": "prajjwal1/bert-tiny",
        "num_classes": 5,
    }
    model = AutoModel.from_config(config)
    assert isinstance(model, MyTransformerSpanClassificationModel)


@pytest.mark.slow
def test_auto_pipeline():
    @dataclass
    class ExampleDocument(TextDocument):
        entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")

    pipeline = AutoPipeline.from_pretrained("pie/example-ner-spanclf-conll03")

    document = ExampleDocument(
        "“Making a super tasty alt-chicken wing is only half of it,” said Po Bronson, general partner at SOSV and managing director of IndieBio."
    )

    pipeline(document, num_workers=0)

    entities = document.entities.predictions
    assert len(entities) == 3
